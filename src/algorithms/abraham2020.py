#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Abraham (2020) [1]_'s algorithm for a single continuous sensitive attribute.

Attributes
----------
ID : str
    Algorithm identifier.
LAMBDA_ : float
    Trade-off between fairness (inf) and utility (0).
MAX_ITER : int
    Maximum number of iterations.
STATS_FORMATTERS : dict[str, str]
    Formatters for printing iteration statistics.

Routine Listings
----------------
compute_fairness_loss()
    Compute fairness loss.
compute_utility_loss()
    Compute utility loss.
estimate_assignment()
    Estimate cluster assignment.
estimate_centroids()
    Estimate cluster centroids.
load()
    Load outputs of Abraham (2020) [1]_'s algorithm.
run()
    Run Abraham (2020) [1]_'s algorithm.

References
----------
.. [1] Savitha Sam Abraham, Deepak P, Sowmya S. Sundaram. `Fairness in
   Clustering with Multiple Sensitive Attributes
   <https://doi.org/10.5441/002/edbt.2020.26>`_. EDBT 2020.

"""


# %% Libraries
import os
import pandas as pd
import shutil
import zipfile
from logging import Logger
from pandas import DataFrame, Series
from tabulate import tabulate

from . import lloyd, utils, SAVE_DIR
from .. import logger
from ..timer import Timer


# %% Algorithm identifier
ID: str = 'Abraham2020'


# %% Parameters
LAMBDA_: float = 1     # trade-off between fairness (inf) and utility (0)
MAX_ITER: int = 100    # maximum number of iterations


# %% Formatters for printing iteration statistics
STATS_FORMATTERS: dict[str, str] = {
    'utility loss': lambda x: f"{x:>14.5e}",
    'fairness loss': lambda x: f"{x:>15.5e}",
    'objective': lambda x: f"{x:>11.5e}",
    'reassignments': lambda x: f"{x:>15}",
    'centroids time': lambda x: f"{x:>16}",
    'assignment time': lambda x: f"{x:>17}",
    'iteration time': lambda x: f"{x:>16}"
    }


# %% Compute fairness loss
def compute_fairness_loss(s: Series, c: Series, n_clusters: int) -> float:
    r"""
    Compute fairness loss.

    This is the modified Abraham (2020) [1]_'s deviation.

    Parameters
    ----------
    s : Series
        Sensitive attribute values.
    c : Series
        Cluster assignment.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    float
        Fairness loss.

    Notes
    -----
    Abraham (2020, Equation 22) [1]_'s deviation for the cluster
    assignment :math:`\mathcal{C}` given a single continuous sensitive
    attribute :math:`\mathcal{S}` is computed as

    .. math::
        \sum_{C \in \mathcal{C}}\
        \bigg(\frac{|C|}{|X|}\bigg)^2
        \big(C.\mathcal{S} - X.\mathcal{S}\big)^2

    where :math:`X.\mathcal{S}` and :math:`C.\mathcal{S}` are
    respectively the average values of the sensitive attribute across
    objects in the dataset :math:`X` and cluster :math:`C`. Towards
    having a similar weighting to the :math:`k`-means objective,
    Abraham (2020, §5.4) suggests a multiplier
    :math:`\frac{{|X|}^2}{{|\mathcal{C}|}^2}`. As the :math:`k`-means
    objective in this codebase is divided by :math:`|X|`, we also
    divide this multiplier by :math:`|X|`, thus giving us

    .. math::
        \frac{|X|}{{|\mathcal{C}|}^2}\
        \sum_{C \in \mathcal{C}}\
        \bigg(\frac{|C|}{|X|}\bigg)^2
        \big(C.\mathcal{S} - X.\mathcal{S}\big)^2

    Now, the range of values that :math:`\big(C.\mathcal{S} -
    X.\mathcal{S}\big)` takes is not constrained and hence places no
    upper bound on the above formulation. To address this,
    :math:`x.\mathcal{S}` is first standardised over the dataset to
    have zero mean and unit variance. The modified Abraham (2020)'s
    deviation is thus

    .. math::
        \frac{1}{{|\mathcal{C}|}^2 \times |X|}\
        \sum_{C \in \mathcal{C}}\
        \bigg(\sum_{x \in C} x.\mathcal{R}\bigg)^2

    where :math:`x.\mathcal{R}` corresponds to the value of
    :math:`x.\mathcal{S}` standardised over the dataset :math:`X`.
    The modified Abraham (2020)'s deviation is used as the fairness
    loss.

    References
    ----------
    .. [1] Savitha Sam Abraham, Deepak P, Sowmya S. Sundaram. `Fairness
       in Clustering with Multiple Sensitive Attributes
       <https://doi.org/10.5441/002/edbt.2020.26>`_. EDBT 2020.

    """
    r = utils.standardise(ser=s)
    n_objects = len(r)
    cluster_losses = r.groupby(c, sort=False).sum() ** 2
    return cluster_losses.sum() / (n_clusters**2 * n_objects)


# %% Compute utility loss
def compute_utility_loss(
        X: DataFrame, n_nonsensitive: int, c: Series,
        centroids: DataFrame
        ) -> float:
    r"""
    Compute utility loss.

    This is the :math:`k`-means objective.

    Parameters
    ----------
    X : DataFrame
        Dataset.
    n_nonsensitive : int
        Number of non-sensitive attributes.
    c : Series
        Cluster assignment.
    centroids : DataFrame
        Centroids.

    Returns
    -------
    float
        Utility loss.

    """
    return lloyd.compute_objective(
        X=X, n_nonsensitive=n_nonsensitive, c=c, centroids=centroids
        )


# %% Compute change in fairness loss on reassigning an object
def _compute_diff_fairness_loss(
        R: float, C: int, r_sums: Series, n_objects: int,
        n_clusters: int
        ) -> Series:
    r"""
    Compute change in fairness loss on reassigning an object.

    Parameters
    ----------
    R : float
        The object's standardised value for the sensitive attribute.
    C : int
        The object's current cluster.
    r_sums : Series
        Sum of objects' standardised values for the sensitive attribute
        in each cluster.
    n_objects : int
        Number of objects in the dataset.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    Series
        Change in fairness loss on reassigning to each cluster.

    Notes
    -----
    If an object :math:`x'` is reassigned from a cluster :math:`C_1` to
    another cluster :math:`C_2 (\neq C_1)`, the change in the fairness
    loss is

    .. math::
        \frac{2}{\textit{n_clusters}^2 \times \textit{n_objects}}
        \times
        x'.\mathcal{R}
        \times
        \bigg(
            \sum_{x \in C_2} x.\mathcal{R}
            -
            \sum_{x \in C_1} x.\mathcal{R}
            +
            x'.\mathcal{R}
        \bigg)

    where :math:`x.\mathcal{R}` corresponds to :math:`x`'s value for
    the sensitive attribute standardised over the dataset :math:`X`.

    """
    # compute multiplier as `R` * 2 / (k^2 * |X|)
    multiplier = R * 2 / (n_clusters**2 * n_objects)
    # compute change in fairness losses of each cluster
    diff_fairness_loss = (r_sums + (R - r_sums.loc[C])) * multiplier
    diff_fairness_loss.loc[C] = 0
    return diff_fairness_loss


# %% Compute change in utility loss on reassigning an object
def _compute_diff_utility_loss(
        C: int, d: Series, n_objects: int, n_nonsensitive: int
        ) -> Series:
    r"""
    Compute change in utility loss on reassigning an object.

    Parameters
    ----------
    C : int
        The object's current cluster.
    d : Series
        The object's distances from all centroids.
    n_objects : int
        Number of objects in the dataset.
    n_nonsensitive : int
        Number of non-sensitive attributes.

    Returns
    -------
    Series
        Change in utility loss on reassigning to each cluster.

    Notes
    -----
    If an object :math:`x'` is reassigned from a cluster :math:`C_1` to
    another cluster :math:`C_2`, the change in the utility loss is

    .. math::
        \frac{1}{\textit{n_nonsensitive} \times \textit{n_objects}}
        \times
        \big(\textit{dist}(x', C_2) - \textit{dist}(x', C_1)\big)

    where :math:`\textit{dist}(x', C)` is the distance of :math:`x'` from
    :math:`C`'s centroid.

    """
    return (d - d.loc[C]) / (n_objects * n_nonsensitive)


# %% Compute change in losses on reassigning an object
def _compute_diffs(
        x: int, X: DataFrame, n_nonsensitive: int, s: Series,
        c: Series, centroids: DataFrame, n_clusters: int,
        lambda_: float
        ) -> tuple[Series, Series, Series]:
    r"""
    Compute change in losses on reassigning an object.

    Parameters
    ----------
    x : int
        The object.
    X : DataFrame
        Dataset.
    n_nonsensitive : int
        Number of non-sensitive attributes.
    s : Series
        Sensitive attribute values.
    c : Series
        Cluster assignment.
    centroids : DataFrame
        Centroids.
    n_clusters : int
        Number of clusters.
    lambda_ : float
        Trade-off between fairness (inf) and utility (0).

    Returns
    -------
    diff_utility_loss : Series
        Change in utility loss on reassigning to each cluster.
    diff_fairness_loss : Series
        Change in fairness loss on reassigning to each cluster.
    diff_objective : Series
        Change in objective on reassigning to each cluster.

    """
    C = c.loc[x]
    utility_loss = dict()
    fairness_loss = dict()
    objective = dict()
    for new_C in range(n_clusters):
        c.loc[x] = new_C
        utility_loss[new_C] = compute_utility_loss(
            X=X, n_nonsensitive=n_nonsensitive, c=c,
            centroids=centroids
            )
        fairness_loss[new_C] = compute_fairness_loss(
            s=s, c=c, n_clusters=n_clusters
            )
        objective[new_C] = utility_loss[new_C] + lambda_*fairness_loss[new_C]
        c.loc[x] = C
    utility_loss = Series(utility_loss)
    fairness_loss = Series(fairness_loss)
    objective = Series(objective)
    diff_utility_loss = utility_loss - utility_loss.loc[C]
    diff_fairness_loss = fairness_loss - fairness_loss.loc[C]
    diff_objective = objective - objective.loc[C]
    return diff_utility_loss, diff_fairness_loss, diff_objective


# %% Estimate cluster assignment
def estimate_assignment(
        X: DataFrame, n_nonsensitive: int, s: Series, c: Series,
        centroids: DataFrame, n_clusters: int, lambda_: float
        ) -> tuple[Series, int]:
    r"""
    Estimate cluster assignment.

    Parameters
    ----------
    X : DataFrame
        Dataset.
    n_nonsensitive : int
        Number of non-sensitive attributes.
    s : Series
        Sensitive attribute values.
    c : Series
        Current cluster assignment.
    centroids : DataFrame
        Centroids.
    n_clusters : int
        Number of clusters.
    lambda_ : float
        Trade-off between fairness (inf) and utility (0).

    Returns
    -------
    Series
        New cluster assignment.
    int
        Elapsed time in nanoseconds.

    Notes
    -----
    Each object in the dataset is greedily reassigned at most once to
    a cluster that minimise the objective. In other words, an object
    :math:`x'` is reassigned from cluster :math:`C_1` to cluster
    :math:`C_2` thus changing the current cluster assignment
    :math:`\mathcal{C}` to :math:`\mathcal{C}'` if

    .. math::
        J(\mathcal{C}', M_{\mathcal{C}'})
        <
        J(\mathcal{C}, M_{\mathcal{C}})

    where :math:`J` is the objective for the given cluster assignment
    :math:`\mathcal{C}` and set of centroids :math:`M_\mathcal{C}`
    given by

    .. math::
        J(\mathcal{C}, M_{\mathcal{C}})
        =
        J_U(\mathcal{C}, M_{\mathcal{C}})
        +
        \lambda J_F(\mathcal{C})

    where :math:`J_U` and :math:`J_F` are respectively the utility loss
    and fairness loss, and :math:`\lambda` controls the trade-off
    between the fairness and utility.

    """
    assert 0 < n_nonsensitive <= len(X.columns)
    assert s.index.equals(X.index)
    assert c.index.equals(X.index)
    assert n_clusters > 0
    assert set(c.unique()) == set(centroids.index) == set(range(n_clusters))
    assert centroids.columns.equals(X.columns)
    assert not centroids.duplicated().any()
    assert lambda_ >= 0

    timer = Timer()
    timer.resume()

    distances = utils.compute_distances(X=X, centroids=centroids)
    n_objects = len(X)
    r = utils.standardise(ser=s)
    # get sums of sensitive attribute's values in each cluster
    r_sums = r.groupby(c, sort=False).sum().sort_index()
    new_c = c.copy()

    # Reorder objects in `distances` so that the ones with least
    # maximum distances are seen first. This way, the objects with
    # higher maximum distances would likely remain in their nearest
    # clusters. This step is NOT done in Abraham (2020).
    reordered_index = distances.max(axis='columns').sort_values().index
    reordered_distances = distances.reindex(index=reordered_index)

    # reassign objects to clusters such that the objective decreases
    for x, d in reordered_distances.iterrows():
        R = r.loc[x]
        C = new_c.loc[x]
        # compute change in utility loss on reassigning `x`
        diff_utility_loss = _compute_diff_utility_loss(
            C=C, d=d, n_objects=n_objects,
            n_nonsensitive=n_nonsensitive
            )
        # compute change in fairness loss on reassigning `x`
        diff_fairness_loss = _compute_diff_fairness_loss(
            R=R, C=C, r_sums=r_sums, n_objects=n_objects,
            n_clusters=n_clusters
            )
        # compute resulting change in objective on reassigning `x`
        diff_objective = diff_utility_loss.add(lambda_ * diff_fairness_loss)

        # # Uncomment this block only if you want to check the correctness
        # timer.pause()
        # u2, f2, o2 = _compute_diffs(
        #     x=x, X=X, n_nonsensitive=n_nonsensitive, s=s, c=new_c,
        #     centroids=centroids, n_clusters=n_clusters, lambda_=lambda_
        #     )
        # assert diff_utility_loss.subtract(u2).abs().sum() < 1e-15
        # assert diff_fairness_loss.subtract(f2).abs().sum() < 1e-14
        # assert diff_objective.subtract(o2).abs().sum() < 1e-14
        # timer.resume()
        # # Block ends here

        # best cluster is one where the difference in the new objective
        # and old objective is the largest negative number
        best_C = diff_objective.idxmin()
        if best_C != C:
            new_c.loc[x] = best_C
            r_sums.loc[C] -= R
            r_sums.loc[best_C] += R
            timer.pause()
            assert set(new_c.unique()) == set(centroids.index), "Some clusters are empty"
            timer.resume()

    timer.pause()

    assert set(new_c.unique()) == set(centroids.index), "Some clusters are empty"
    return new_c, timer.elapsed


# %% Estimate centroids
def estimate_centroids(X: DataFrame, c: Series) -> tuple[DataFrame, int]:
    r"""
    Estimate cluster centroids.

    Parameters
    ----------
    X : DataFrame
        Dataset.
    c : Series
        Cluster assignment.

    Returns
    -------
    DataFrame
        New cluster centroids.
    int
        Elapsed time in nanoseconds.

    Notes
    -----
    For each cluster :math:`C`, its centroid :math:`\mu_C` is estimated
    as

    .. math:: \mu_C = \frac{1}{|C|} \sum_{x \in C} x

    """
    new_centroids, centroids_time = lloyd.estimate_centroids(X=X, c=c)
    return new_centroids, centroids_time


# %% Load algorithm outputs
def load(
        n_clusters: int, dataset_name: str, init_method: str,
        random_state: int, iter_: int = -1
        ) -> tuple[Series, DataFrame, DataFrame, DataFrame]:
    r"""
    Load outputs of Abraham (2020) [1]_'s algorithm.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    dataset_name : str
        Name of the dataset.
    init_method : str
        Method used for centroid initialisation.
    random_state : int
        Random state used for centroid initialisation.
    iter_ : int, optional
        Load cluster assignment and centroids output at this iteration.
        -1 corresponds to the last iteration. The default is -1.

    Returns
    -------
    c : Series
        Cluster assignment.
    centroids : DataFrame
        Centroids.
    info : DataFrame
        Information summary of the run.
    stats : DataFrame
        Statistics of each iteration in the run.

    References
    ----------
    .. [1] Savitha Sam Abraham, Deepak P, Sowmya S. Sundaram. `Fairness
       in Clustering with Multiple Sensitive Attributes
       <https://doi.org/10.5441/002/edbt.2020.26>`_. EDBT 2020.

    """
    assert n_clusters > 0
    assert iter_ == -1 or iter_ > 0
    load_dir = os.path.join(
        SAVE_DIR, dataset_name, f'k={n_clusters}', init_method,
        f'r={random_state}', ID
        )
    print(f"Loading Abraham (2020)'s clusters and centroids from '{load_dir}'")
    if iter_ == -1:
        iter_ = utils.get_last_iteration(save_dir=load_dir)
    c_path = os.path.join(load_dir, 'clusters.zip')
    centroids_path = os.path.join(load_dir, 'centroids.zip')
    with zipfile.ZipFile(c_path) as zf:
        c = pd.read_csv(zf.open(f'{iter_}.csv'), index_col=0).squeeze()
    assert isinstance(c, Series)
    with zipfile.ZipFile(centroids_path) as zf:
        centroids = pd.read_csv(zf.open(f'{iter_}.csv'), index_col=0)
    info_path = os.path.join(load_dir, 'info.csv')
    info = pd.read_csv(
        info_path,
        index_col=[
            'dataset', 'sensitive_attribute', 'n_clusters',
            'init_method', 'random_state', 'algorithm', 'lambda_',
            'max_iter'
            ]
        )
    stats_path = os.path.join(load_dir, 'stats.csv')
    stats = pd.read_csv(stats_path, index_col='iter')
    stats['reassignments'] = stats['reassignments'].transform(
        lambda x: x if x == '-' else int(x)
        )
    stats['centroids time'] = stats['centroids time'].transform(
        lambda x: x if x == '-' else int(x)
        )
    stats['assignment time'] = stats['assignment time'].transform(
        lambda x: x if x == '-' else int(x)
        )
    return c, centroids, info, stats


# %% Compute and show iteration statistics
def _show_iter_stats(
        X: DataFrame, n_nonsensitive: int, s: Series, c: Series,
        centroids: DataFrame, n_clusters: int, lambda_: float, i: int,
        n_reassignments: int | str, centroids_time: int | str,
        c_time: int | str, iter_time: int, lgr: Logger
        ) -> dict[str, int | float | str]:
    r"""
    Compute and show iteration statistics.

    Parameters
    ----------
    X : DataFrame
        Dataset.
    n_nonsensitive : int
        Number of non-sensitive attributes.
    s : Series
        Sensitive attribute values.
    c : Series
        Cluster assignment.
    centroids : DataFrame
        Centroids.
    n_clusters : int
        Number of clusters.
    lambda_ : float
        Trade-off between fairness (inf) and utility (0).
    i : int
        Iteration number.
    n_reassignments : int | str
        Number of cluster reassignments. '-' if the cluster assignment
        was not estimated in this iteration.
    centroids_time : int | str
        Elapsed time in nanoseconds for centroid estimation. '-' if
        this is the first iteration.
    c_time : int | str
        Elapsed time in nanoseconds for cluster assignment estimation.
        '-' if the cluster assignment was not estimated in this
        iteration.
    iter_time : int
        Elapsed time in nanoseconds for this iteration.
    lgr : Logger
        Logger.

    Returns
    -------
    dict[str, int | float | str]
        Statistics for this iteration.

    """
    utility_loss = compute_utility_loss(
        X=X, n_nonsensitive=n_nonsensitive, c=c, centroids=centroids
        )
    fairness_loss = compute_fairness_loss(s=s, c=c, n_clusters=n_clusters)
    objective = utility_loss + lambda_*fairness_loss
    iter_stats = {
        'utility loss': utility_loss,
        'fairness loss': fairness_loss,
        'objective': objective,
        'reassignments': n_reassignments,
        'centroids time': centroids_time,
        'assignment time': c_time,
        'iteration time': iter_time
        }
    str_list = [f"{i:6}"]
    str_list.extend(
        fn(iter_stats[k]) for k, fn in STATS_FORMATTERS.items()
        )
    lgr.info("│ " + " │ ".join(str_list) + " │")
    return iter_stats


# %% Run Abraham (2020)'s algorithm for a single continuous sensitive attribute
def run(
        X: DataFrame, n_nonsensitive: int, s: Series, n_clusters: int,
        init_centroids: DataFrame, dataset_name: str, init_method: str,
        random_state: int, lambda_: float = LAMBDA_,
        max_iter: int = MAX_ITER, export: bool = False
        ) -> tuple[Series, DataFrame, DataFrame, DataFrame]:
    r"""
    Run Abraham (2020) [1]_'s algorithm.

    This function is for a single continuous sensitive attribute.

    Parameters
    ----------
    X : DataFrame
        Dataset.
    n_nonsensitive : int
        Number of non-sensitive attributes.
    s : Series
        Sensitive attribute values.
    n_clusters : int
        Number of clusters.
    init_centroids : DataFrame
        Initialised centroids.
    dataset_name : str
        Name of the dataset.
    init_method : str
        Method used for centroid initialisation.
    random_state : int
        Random state used for centroid initialisation.
    lambda_ : float, optional
        Trade-off between fairness (inf) and utility (0). The default
        is LAMBDA_.
    max_iter : int, optional
        Maximum number of iterations. The default is MAX_ITER.
    export : bool, optional
        Whether to export outputs. The default is False.

    Returns
    -------
    c : Series
        Cluster assignment.
    centroids : DataFrame
        Centroids.
    info : DataFrame
        Information summary of the run.
    stats : DataFrame
        Statistics of each iteration in the run.

    References
    ----------
    .. [1] Savitha Sam Abraham, Deepak P, Sowmya S. Sundaram. `Fairness
       in Clustering with Multiple Sensitive Attributes
       <https://doi.org/10.5441/002/edbt.2020.26>`_. EDBT 2020.

    """
    assert 0 < n_nonsensitive <= len(X.columns)
    assert s.index.equals(X.index)
    assert n_clusters > 0
    assert set(init_centroids.index) == set(range(n_clusters))
    assert init_centroids.columns.equals(X.columns)
    assert not init_centroids.duplicated().any()
    assert lambda_ >= 0
    assert max_iter >= 1

    if export:
        save_dir = os.path.join(
            SAVE_DIR, dataset_name, f'k={n_clusters}', init_method,
            f'r={random_state}', ID
            )
        os.makedirs(save_dir, exist_ok=False)
        lgr = logger.get(log_dir=save_dir)
    else:
        lgr = logger.get(name=ID)

    lgr.info("Initiating Abraham (2020)'s algorithm")

    info = {
        'dataset': [dataset_name],
        'sensitive_attribute': [s.name],
        'n_clusters': [n_clusters],
        'init_method': [init_method],
        'random_state': [random_state],
        'algorithm': [ID],
        'lambda_': [lambda_],
        'max_iter': [max_iter]
        }

    info_str = tabulate(
        info,
        headers='keys',
        tablefmt='rounded_outline',
        stralign='right'
        )
    for line in info_str.split('\n'):
        lgr.info(line)

    if export:
        lgr.info(f"Exporting clusters and centroids to '{save_dir}'")
        c_dir = os.path.join(save_dir, 'clusters')
        centroids_dir = os.path.join(save_dir, 'centroids')
        os.makedirs(c_dir, exist_ok=False)
        os.makedirs(centroids_dir, exist_ok=False)

    algo_timer = Timer()
    iter_timer = Timer()
    stats = dict()

    # run algorithm
    lgr.info("Running algorithm")
    lgr.info("╭────────┬────────────────┬─────────────────┬─────────────┬─────────────────┬──────────────────┬───────────────────┬──────────────────╮")
    lgr.info("│   iter │   utility loss │   fairness loss │   objective │   reassignments │   centroids time │   assignment time │   iteration time │")
    lgr.info("├────────┼────────────────┼─────────────────┼─────────────┼─────────────────┼──────────────────┼───────────────────┼──────────────────┤")

    # initialise centroids
    centroids = init_centroids
    if export:
        centroids.to_csv(os.path.join(centroids_dir, '1.csv'))
    # estimate cluster assignment
    c, c_time = lloyd.estimate_assignment(X=X, centroids=centroids)
    Timer.add_to_all(iter_timer, algo_timer, ns=c_time)
    if export:
        c.to_csv(os.path.join(c_dir, '1.csv'))
    # show iteration stats
    stats[1] = _show_iter_stats(
        X=X, n_nonsensitive=n_nonsensitive, s=s, c=c,
        centroids=centroids, n_clusters=n_clusters, lambda_=lambda_,
        i=1, n_reassignments=len(c), centroids_time='-', c_time=c_time,
        iter_time=iter_timer.elapsed, lgr=lgr
        )
    Timer.reset_all(iter_timer)

    Timer.resume_all(algo_timer, iter_timer)
    converged = False
    for i in range(2, max_iter):
        Timer.pause_all(iter_timer, algo_timer)
        # estimate centroids
        new_centroids, centroids_time = estimate_centroids(X=X, c=c)
        assert new_centroids.index.equals(centroids.index)
        assert new_centroids.columns.equals(centroids.columns)
        Timer.add_to_all(iter_timer, algo_timer, ns=centroids_time)
        if export:
            new_centroids.to_csv(os.path.join(centroids_dir, f'{i}.csv'))
        # estimate cluster assignment
        new_c, c_time = estimate_assignment(
            X=X, n_nonsensitive=n_nonsensitive, s=s, c=c,
            centroids=new_centroids, n_clusters=n_clusters,
            lambda_=lambda_
            )
        Timer.add_to_all(iter_timer, algo_timer, ns=c_time)
        if export:
            new_c.to_csv(os.path.join(c_dir, f'{i}.csv'))
        Timer.resume_all(algo_timer, iter_timer)
        # compute convergence condition
        n_reassignments = sum(~new_c.eq(c))
        # update clustering state
        centroids = new_centroids
        c = new_c
        Timer.pause_all(iter_timer, algo_timer)
        # show iteration stats
        stats[i] = _show_iter_stats(
            X=X, n_nonsensitive=n_nonsensitive, s=s, c=c,
            centroids=centroids, n_clusters=n_clusters,
            lambda_=lambda_, i=i, n_reassignments=n_reassignments,
            centroids_time=centroids_time, c_time=c_time,
            iter_time=iter_timer.elapsed, lgr=lgr
            )
        Timer.reset_all(iter_timer)

        Timer.resume_all(algo_timer)
        # stop if cluster assignment has not changed
        if n_reassignments == 0:
            converged = True
            Timer.pause_all(algo_timer)
            info['converged'] = True
            info['how_terminated'] = 'strict convergence'
            Timer.resume_all(algo_timer, iter_timer)
            break

        Timer.resume_all(iter_timer)

    if converged:
        Timer.pause_all(iter_timer, algo_timer)
        Timer.reset_all(iter_timer)

    else:
        i = i + 1
        Timer.pause_all(iter_timer, algo_timer)
        # estimate centroids
        new_centroids, centroids_time = estimate_centroids(X=X, c=c)
        assert new_centroids.index.equals(centroids.index)
        assert new_centroids.columns.equals(centroids.columns)
        Timer.add_to_all(iter_timer, algo_timer, ns=centroids_time)
        if export:
            new_centroids.to_csv(os.path.join(centroids_dir, f'{i}.csv'))
            c.to_csv(os.path.join(c_dir, f'{i}.csv'))
        Timer.resume_all(algo_timer, iter_timer)
        # update clustering state
        centroids = new_centroids
        Timer.pause_all(iter_timer, algo_timer)
        # show iteration stats
        stats[i] = _show_iter_stats(
            X=X, n_nonsensitive=n_nonsensitive, s=s, c=c,
            centroids=centroids, n_clusters=n_clusters,
            lambda_=lambda_, i=i, n_reassignments='-',
            centroids_time=centroids_time, c_time='-',
            iter_time=iter_timer.elapsed, lgr=lgr
            )
        Timer.reset_all(iter_timer)

        info['converged'] = False
        info['how_terminated'] = 'maximum number of iterations'

    lgr.info("╰────────┴────────────────┴─────────────────┴─────────────┴─────────────────┴──────────────────┴───────────────────┴──────────────────╯")
    if info['converged']:
        lgr.info(f"Converged at iteration {i}: {info['how_terminated']}")
    else:
        lgr.info(f"Terminated at iteration {i}: {info['how_terminated']}")
    lgr.info(f"Algorithm running time: {algo_timer.elapsed * 1e-9} seconds")

    if export:
        _ = shutil.make_archive(base_name=c_dir, format='zip', root_dir=c_dir)
        _ = shutil.make_archive(
            base_name=centroids_dir, format='zip', root_dir=centroids_dir
            )
        shutil.rmtree(c_dir, ignore_errors=True)
        shutil.rmtree(centroids_dir, ignore_errors=True)

    logger.shutdown(lgr=lgr)

    info['n_iter'] = [i]                                   # number of iterations elapsed
    info['utility loss'] = [stats[i]['utility loss']]      # utility loss after last iteration
    info['fairness loss'] = [stats[i]['fairness loss']]    # fairness loss after last iteration
    info['objective'] = [stats[i]['objective']]            # objective value after last iteration
    info['algo_time_ns'] = [algo_timer.elapsed]            # algorithm running time in nanoseconds
    info = DataFrame.from_dict(info)
    index_col = [
        'dataset', 'sensitive_attribute', 'n_clusters', 'init_method',
        'random_state', 'algorithm', 'lambda_', 'max_iter'
        ]
    info = info.set_index(index_col)
    stats = DataFrame.from_dict(stats, orient='index')
    stats.index.name = 'iter'

    if export:
        info.to_csv(os.path.join(save_dir, 'info.csv'))
        stats.to_csv(os.path.join(save_dir, 'stats.csv'))

    return c, centroids, info, stats


# %% END OF FILE
