#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Lloyd's heuristic [1]_ for :math:`k`-means clustering.

Attributes
----------
ID : str
    Algorithm identifier.
MAX_ITER : int
    Maximum number of iterations.
STATS_FORMATTERS : dict[str, str]
    Formatters for printing iteration statistics.
TOL : float
    Relative tolerance to declare convergence.

Routine Listings
----------------
compute_objective()
    Compute :math:`k`-means objective.
estimate_assignment()
    Estimate cluster assignment.
estimate_centroids()
    Estimate cluster centroids.
load()
    Load outputs of Lloyd's heuristic [1]_ for :math:`k`-means clustering.
run()
    Run Lloyd's heuristic [1]_ for :math:`k`-means clustering.

References
----------
.. [1] Stuart P Lloyd. `Least squares quantization in PCM
   <https://doi.org/10.1109/TIT.1982.1056489>`_. IEEE Transactions on
   Information Theory, 28(2). 1982.

"""


# %% Libraries
import os
import pandas as pd
import shutil
import zipfile
from logging import Logger
from pandas import DataFrame, Series
from sklearn.cluster import _kmeans
from tabulate import tabulate

from . import utils, SAVE_DIR
from .. import logger
from ..timer import Timer


# %% Algorithm identifier
ID: str = 'Lloyd'


# %% Parameters
MAX_ITER: int = 100    # maximum number of iterations
TOL: float = 1e-4      # relative tolerance to declare convergence


# %% Formatters for printing iteration statistics
STATS_FORMATTERS: dict[str, str] = {
    'objective': lambda x: f"{x:>11.5e}",
    'reassignments': lambda x: f"{x:>15}",
    'centroid shift': lambda x: f"{x:>16}" if x == '-' else f"{x:>16.5e}",
    'centroids time': lambda x: f"{x:>16}",
    'assignment time': lambda x: f"{x:>17}",
    'iteration time': lambda x: f"{x:>16}"
    }


# %% Compute objective
def compute_objective(
        X: DataFrame, n_nonsensitive: int, c: Series,
        centroids: DataFrame
        ) -> float:
    r"""
    Compute :math:`k`-means objective.

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
        :math:`k`-means objective.

    Notes
    -----
    The :math:`k`-means objective is computed as

    .. math::
        \frac{1}{\textit{n_nonsensitive} \times |X|}
        \times \sum_{C \in \mathcal{C}}\ \sum_{x \in C}\ {||x - \mu_C||}^2

    where :math:`\mathcal{C}` is the cluster assignment and
    :math:`\mu_C` is the centroid of cluster :math:`C`.

    """
    distances = utils.compute_distances(X=X, centroids=centroids)
    object_losses = utils.get_object_losses(distances=distances, c=c)
    return object_losses.mean() / n_nonsensitive


# %% Estimate cluster assignment
def estimate_assignment(
        X: DataFrame, centroids: DataFrame
        ) -> tuple[Series, int]:
    r"""
    Estimate cluster assignment.

    Parameters
    ----------
    X : DataFrame
        Dataset.
    centroids : DataFrame
        Centroids.

    Returns
    -------
    Series
        New cluster assignment.
    int
        Elapsed time in nanoseconds.

    Notes
    -----
    This is the E-step of the algorithm. An object :math:`x` is
    assigned to the cluster whose centroid is the nearest, *i.e.*,
    :math:`x` is assigned to

    .. math::
        \underset{C \in \mathcal{C}}{\operatorname{argmin}}
        {||x - \mu_C||}^2

    where :math:`\mathcal{C}` is the cluster assignment and
    :math:`\mu_C` is the centroid of cluster :math:`C`.

    """
    assert set(centroids.index) == set(range(len(centroids)))
    assert centroids.columns.equals(X.columns)
    assert not centroids.duplicated().any()
    timer = Timer()
    timer.resume()
    distances = utils.compute_distances(X=X, centroids=centroids)
    new_c = distances.idxmin(axis='columns')
    new_c.name = distances.columns.name
    timer.pause()
    assert new_c.index.equals(X.index)
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
    This is the M-step of the algorithm. For each cluster :math:`C`,
    its centroid :math:`\mu_C` is estimated as

    .. math:: \mu_C = \frac{1}{|C|} \sum_{x \in C} x

    """
    assert c.index.equals(X.index)
    timer = Timer()
    timer.resume()
    new_centroids = X.groupby(c, sort=False).mean().sort_index()
    timer.pause()
    return new_centroids, timer.elapsed


# %% Load algorithm outputs
def load(
        n_clusters: int, dataset_name: str, init_method: str,
        random_state: int, iter_: int = -1
        ) -> tuple[Series, DataFrame, DataFrame, DataFrame]:
    r"""
    Load outputs of Lloyd's heuristic [1]_ for :math:`k`-means clustering.

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
    .. [1] Stuart P Lloyd. `Least squares quantization in PCM
       <https://doi.org/10.1109/TIT.1982.1056489>`_. IEEE Transactions
       on Information Theory, 28(2). 1982.

    """
    assert n_clusters > 0
    assert iter_ == -1 or iter_ > 0
    load_dir = os.path.join(
        SAVE_DIR, dataset_name, f'k={n_clusters}', init_method,
        f'r={random_state}', ID
        )
    print(f"Loading Lloyd clusters and centroids from '{load_dir}'")
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
            'dataset', 'n_clusters', 'init_method', 'random_state',
            'algorithm', 'max_iter', 'tol'
            ]
        )
    stats_path = os.path.join(load_dir, 'stats.csv')
    stats = pd.read_csv(stats_path, index_col='iter')
    stats['reassignments'] = stats['reassignments'].transform(
        lambda x: x if x == '-' else int(x)
        )
    stats['centroid shift'] = stats['centroid shift'].transform(
        lambda x: x if x == '-' else float(x)
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
        X: DataFrame, n_nonsensitive: int, c: Series,
        centroids: DataFrame, i: int, n_reassignments: int | str,
        centroid_shift: float | str, centroids_time: int | str,
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
    c : Series
        Cluster assignment.
    centroids : DataFrame
        Centroids.
    i : int
        Iteration number.
    n_reassignments : int | str
        Number of cluster reassignments. '-' if the cluster assignment
        was not estimated in this iteration.
    centroid_shift : float | str
        Frobenius norm of the difference in new and old centroids. '-'
        if this is the first iteration.
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
    objective = compute_objective(
        X=X, n_nonsensitive=n_nonsensitive, c=c, centroids=centroids
        )
    iter_stats = {
        'objective': objective,
        'reassignments': n_reassignments,
        'centroid shift': centroid_shift,
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


# %% Run Lloyd's heuristic
def run(
        X: DataFrame, n_nonsensitive: int, n_clusters: int,
        init_centroids: DataFrame, dataset_name: str, init_method: str,
        random_state: int, max_iter: int = MAX_ITER, tol: float = TOL,
        export: bool = False
        ) -> tuple[Series, DataFrame, DataFrame, DataFrame]:
    r"""
    Run Lloyd's heuristic [1]_ for :math:`k`-means clustering.

    Parameters
    ----------
    X : DataFrame
        Dataset.
    n_nonsensitive : int
        Number of non-sensitive attributes.
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
    max_iter : int, optional
        Maximum number of iterations. The default is MAX_ITER.
    tol : float, optional
        Relative tolerance to declare convergence. The default is TOL.
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
    .. [1] Stuart P Lloyd. `Least squares quantization in PCM
       <https://doi.org/10.1109/TIT.1982.1056489>`_. IEEE Transactions
       on Information Theory, 28(2). 1982.

    """
    assert 0 < n_nonsensitive <= len(X.columns)
    assert n_clusters > 0
    assert set(init_centroids.index) == set(range(n_clusters))
    assert init_centroids.columns.equals(X.columns)
    assert not init_centroids.duplicated().any()
    assert max_iter >= 1
    assert tol > 0

    if export:
        save_dir = os.path.join(
            SAVE_DIR, dataset_name, f'k={n_clusters}', init_method,
            f'r={random_state}', ID
            )
        os.makedirs(save_dir, exist_ok=False)
        lgr = logger.get(log_dir=save_dir)
    else:
        lgr = logger.get(name=ID)

    lgr.info("Initiating Lloyd's heuristic")

    info = {
        'dataset': [dataset_name],
        'n_clusters': [n_clusters],
        'init_method': [init_method],
        'random_state': [random_state],
        'algorithm': [ID],
        'max_iter': [max_iter],
        'tol': [tol]
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

    Timer.resume_all(algo_timer)
    # set tolerance dependent on the dataset (as done in sklearn's kmeans)
    tol_ = _kmeans._tolerance(X=X, tol=tol)
    Timer.pause_all(algo_timer)

    # run algorithm
    lgr.info("Running algorithm")
    lgr.info("╭────────┬─────────────┬─────────────────┬──────────────────┬──────────────────┬───────────────────┬──────────────────╮")
    lgr.info("│   iter │   objective │   reassignments │   centroid shift │   centroids time │   assignment time │   iteration time │")
    lgr.info("├────────┼─────────────┼─────────────────┼──────────────────┼──────────────────┼───────────────────┼──────────────────┤")

    # initialise centroids
    centroids = init_centroids
    if export:
        centroids.to_csv(os.path.join(centroids_dir, '1.csv'))
    # estimate cluster assignment
    c, c_time = estimate_assignment(X=X, centroids=centroids)
    Timer.add_to_all(iter_timer, algo_timer, ns=c_time)
    if export:
        c.to_csv(os.path.join(c_dir, '1.csv'))
    # show iteration stats
    stats[1] = _show_iter_stats(
        X=X, n_nonsensitive=n_nonsensitive, c=c, centroids=centroids,
        i=1, n_reassignments=len(c), centroid_shift='-',
        centroids_time='-', c_time=c_time,
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
        new_c, c_time = estimate_assignment(X=X, centroids=new_centroids)
        Timer.add_to_all(iter_timer, algo_timer, ns=c_time)
        if export:
            new_c.to_csv(os.path.join(c_dir, f'{i}.csv'))
        Timer.resume_all(algo_timer, iter_timer)
        # compute convergence conditions
        n_reassignments = sum(~new_c.eq(c))
        centroid_shift = utils.compute_centroid_shift(
            centroids=centroids, new_centroids=new_centroids
            )
        # update clustering state
        centroids = new_centroids
        c = new_c
        Timer.pause_all(iter_timer, algo_timer)
        # show iteration stats
        stats[i] = _show_iter_stats(
            X=X, n_nonsensitive=n_nonsensitive, c=c,
            centroids=centroids, i=i, n_reassignments=n_reassignments,
            centroid_shift=centroid_shift,
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
        # stop if centroids have not changed enough
        if centroid_shift <= tol_:
            converged = True
            Timer.pause_all(algo_timer)
            info['converged'] = True
            info['how_terminated'] = 'centroid shift within tolerance'
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
        # compute convergence conditions
        centroid_shift = utils.compute_centroid_shift(
            centroids=centroids, new_centroids=new_centroids
            )
        # update clustering state
        centroids = new_centroids
        Timer.pause_all(iter_timer, algo_timer)
        # show iteration stats
        stats[i] = _show_iter_stats(
            X=X, n_nonsensitive=n_nonsensitive, c=c,
            centroids=centroids, i=i, n_reassignments='-',
            centroid_shift=centroid_shift,
            centroids_time=centroids_time, c_time='-',
            iter_time=iter_timer.elapsed, lgr=lgr
            )
        Timer.reset_all(iter_timer)

        info['converged'] = False
        info['how_terminated'] = 'maximum number of iterations'

    lgr.info("╰────────┴─────────────┴─────────────────┴──────────────────┴──────────────────┴───────────────────┴──────────────────╯")
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
    info['objective'] = [stats[i]['objective']]            # objective value after last iteration
    info['algo_time_ns'] = [algo_timer.elapsed]            # algorithm running time in nanoseconds
    info = DataFrame.from_dict(info)
    index_col = [
        'dataset', 'n_clusters', 'init_method', 'random_state',
        'algorithm', 'max_iter', 'tol'
        ]
    info = info.set_index(index_col)
    stats = DataFrame.from_dict(stats, orient='index')
    stats.index.name = 'iter'

    if export:
        info.to_csv(os.path.join(save_dir, 'info.csv'))
        stats.to_csv(os.path.join(save_dir, 'stats.csv'))

    return c, centroids, info, stats


# %% END OF FILE
