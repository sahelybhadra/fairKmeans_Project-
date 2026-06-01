#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
"""


# %% Libraries
import numpy as np
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
import warnings
warnings.filterwarnings('ignore')

# %% Algorithm identifier
ID: str = 'Ziko2021'


# %% Parameters
LAMBDA_: float = 1         # trade-off between fairness (inf) and utility (0)
MAX_ITER: int = 100        # maximum number of iterations
L: float = 2               # Lipschitz-gradient constant
BOUND_ITER: int = 10000    # maximum number of iterations in the cluster assignment step
TOL: float = 1e-5          # tolerance to declare convergence in the cluster assignment step


# %% Formatters for printing iteration statistics
STATS_FORMATTERS: dict[str, str] = {
    'auxiliary fn': lambda x: f"{x:>14}" if x == '-' else f"{x:>14.5e}",
    'utility loss': lambda x: f"{x:>14.5e}",
    'fairness loss': lambda x: f"{x:>15.5e}",
    'objective': lambda x: f"{x:>11.5e}",
    'reassignments': lambda x: f"{x:>15}",
    'centroids time': lambda x: f"{x:>16}",
    'assignment time': lambda x: f"{x:>17}",
    'iteration time': lambda x: f"{x:>16}"
    }


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


# %% Compute fairness loss
def compute_fairness_loss(s: Series, c: Series) -> float:
    r"""
    Compute fairness loss.

    Parameters
    ----------
    s : Series
        Sensitive attribute values.
    c : Series
        Cluster assignment.

    Returns
    -------
    float
        Fairness loss.

    Notes
    -----
    Ziko (2021, Equation 5) [1]_'s fairness loss for the cluster
    assignment :math:`\mathcal{C}` given a continuous sensitive
    attribute :math:`\mathcal{S}` is computed as

    .. math::
        \frac{1}{|\mathcal{C}|}\
        \sum_{C \in \mathcal{C}}\
        \sum_{S \in \mathcal{S}}\
        \big(P_X(S) \log |C| - P_X(S) \log |C \cap S| \big)

    where :math:`P_X` is the sensitive attribute's probability
    distribution in the dataset. On further simplification, this
    evaluates to

    .. math::
        \frac{1}{|\mathcal{C}|}\
        \bigg(
            \sum_{C \in \mathcal{C}}\
            \log |C|
            -
            \sum_{C \in \mathcal{C}}\
            \sum_{S \in \mathcal{S}}\
            P_X(S) \log |C \cap S|
        \bigg)

    References
    ----------
    .. [1] Imtiaz Masud Ziko, Jing Yuan, Eric Granger, Ismail Ben Ayed.
       `Variational Fair Clustering
       <https://doi.org/10.1609/aaai.v35i12.17336>`_. AAAI 2021.

    """
    count_C = c.value_counts(sort=False)
    term1 = np.log(count_C).sum()
    P_X = s.value_counts(normalize=True, sort=False)
    count_CS = s.groupby(c, sort=False).value_counts(sort=False)
    count_CS = count_CS.unstack(level=c.name, fill_value=1e-100)
    term2 = np.nansum(np.log(count_CS).multiply(P_X, axis='index'))
    return (term1 - term2) / len(count_C)


def compute_fairness_loss_new(s: Series, c: Series) -> float:
    count_C = c.value_counts(sort=False)
    term1 = np.log(count_C).sum()
    P_X = s.value_counts(normalize=True, sort=False)


# %% Compute k-means potentials of each object
def _compute_a(
        X: DataFrame, n_nonsensitive: int, centroids: DataFrame
        ) -> DataFrame:
    r"""
    Compute :math:`k`-means potentials of each object.

    Parameters
    ----------
    X : DataFrame
        Dataset.
    n_nonsensitive : int
        Number of non-sensitive attributes.
    centroids : DataFrame
        Centroids.

    Returns
    -------
    DataFrame
        :math:`k`-means potentials of each object.

    Notes
    -----
    The :math:`k`-means potential :math:`a_{x,C}` of an object
    :math:`x` from a centroid :math:`\mu_C` is the object's distance
    from the centroid (Ziko 2021, Table 1) [1]_ divided by the number
    of non-sensitive attributes, computed as

    .. math::
        \frac{1}{\textit{n_nonsensitive}} \times {||x - \mu_C||}^2

    References
    ----------
    .. [1] Imtiaz Masud Ziko, Jing Yuan, Eric Granger, Ismail Ben Ayed.
       `Variational Fair Clustering
       <https://doi.org/10.1609/aaai.v35i12.17336>`_. AAAI 2021.

    """
    distances = utils.compute_distances(X=X, centroids=centroids)
    return distances / n_nonsensitive


# %% Compute b term in auxiliary function for fairness
def _compute_fairness_auxiliary(
        U: Series, V: DataFrame, soft_c: DataFrame, L_: float
        ) -> DataFrame:
    r"""
    Compute the :math:`b` term in auxiliary function for fairness.

    Parameters
    ----------
    U : Series
        Sensitive attribute's probability mass function in the dataset.
    V : DataFrame
        Indicates the sensitive value of each object in the dataset.
    soft_c : DataFrame
        Soft cluster assignment.
    L_ : float
        Lipschitz-gradient constant.

    Returns
    -------
    DataFrame
        Auxiliary function for fairness loss.

    Notes
    -----
    The :math:`b_{x,C}` term of an object :math:`x` with respect to a
    cluster :math:`C` in the auxiliary function for fairness
    (Ziko 2021, Equation 9) [1]_ is computed as

    .. math::
        \frac{1}{\textit{L_}}\
        \sum_{S \in \mathcal{S}}\
        \bigg(\frac{}{} \bigg)

    References
    ----------
    .. [1] Imtiaz Masud Ziko, Jing Yuan, Eric Granger, Ismail Ben Ayed.
       `Variational Fair Clustering
       <https://doi.org/10.1609/aaai.v35i12.17336>`_. AAAI 2021.

    """
    # equivalent to `compute_b_j_parallel` in original code
    # µ_j / S_k
    term_1 = DataFrame(
        np.outer(U, 1/soft_c.sum(axis='index')),
        index=U.index,
        columns=soft_c.columns
        )
    # µ_j * v_j
    term_2_numerator = V.multiply(U, axis='index')
    # V_j * S_k
    term_2_denominator = DataFrame(
        np.dot(V, soft_c),
        index=V.index,
        columns=soft_c.columns
        )
    # clip `term_2_denominator` following `compute_b_j` in original code
    term_2_denominator = np.maximum(term_2_denominator, 1e-100)
    term_2 = term_2_numerator.divide(
        #term_2_denominator.stack(dropna=False, sort=False),
        term_2_denominator.stack(dropna=False),
        axis='index',
        level=U.index.name
        )
    
    U.index.name='Age'
    #print("U")
    #print(U.index.name)
    term_2 = term_2.T.stack(level=U.index.name, dropna=False)
    difference = term_1.subtract(term_2, axis='index', level=U.index.name)
    return difference.groupby(level=V.columns.name, sort=False).sum() / L_


# %% Compute overall auxiliary function
def _compute_overall_auxiliary(
        a: DataFrame, lambda_b: DataFrame, new_soft_c: DataFrame,
        soft_c: DataFrame
        ) -> float:
    # clip `new_soft_c` and `soft_c` following `bound_energy` in original code
    sum_ = (
        a + lambda_b
        + np.log(np.maximum(new_soft_c, 1e-100))
        - np.log(np.maximum(soft_c, 1e-100))
        )
    return np.nansum(new_soft_c.multiply(sum_, axis='index')) / len(a)


# %% Estimate cluster assignment
def estimate_assignment(
        X: DataFrame, n_nonsensitive: int, s: Series,
        centroids: DataFrame, n_clusters: int, lambda_: float,
        L_: float, bound_iter: int, tol: float
        ) -> tuple[Series, float, int]:
    assert 0 < n_nonsensitive <= len(X.columns)
    assert s.index.equals(X.index)
    assert n_clusters > 0
    assert set(centroids.index) == set(range(n_clusters))
    assert centroids.columns.equals(X.columns)
    assert not centroids.duplicated().any()
    assert lambda_ >= 0
    assert 0 < L_ <= len(X)
    assert bound_iter >= 1
    assert tol > 0

    timer = Timer()
    timer.resume()

    V = pd.get_dummies(s, dtype=int).T
    V.index.name = s.name
    U = s.value_counts(normalize=True, sort=False).sort_index()
    lambda_ = lambda_ * len(X) / n_clusters

    # compute k-means potentials of the objects
    a = _compute_a(X=X, n_nonsensitive=n_nonsensitive, centroids=centroids)
    timer.pause()
    assert a.idxmin(axis='columns').nunique() == len(centroids), "Some clusters are empty"
    timer.resume()

    # initialise soft clusters
    soft_c = np.exp(-a)
    soft_c = soft_c.divide(soft_c.sum(axis='columns'), axis='index')

    auxiliary = None
    for _ in range(bound_iter):
        # compute auxiliary function for fairness loss
        b = _compute_fairness_auxiliary(U=U, V=V, soft_c=soft_c, L_=L_)
        lambda_b = lambda_ * b
        # -(a + λb)
        neg_sum = -(a + lambda_b)
        # subtract maximum to prevent overflow due to np.exp
        neg_sum = neg_sum.subtract(neg_sum.max(axis='columns'), axis='index')
        product = soft_c.multiply(np.exp(neg_sum), axis='index')
        # compute new soft clusters using Equation 13 in Ziko (2021)
        new_soft_c = product.divide(product.sum(axis='columns'), axis='index')
        # compute overall auxiliary function
        new_auxiliary = _compute_overall_auxiliary(
            a=a, lambda_b=lambda_b, new_soft_c=new_soft_c,
            soft_c=soft_c
            )
        if auxiliary is not None:
            # stop if overall auxiliary function is diverging
            if new_auxiliary > auxiliary:
                # fall back to old cluster assignment and stop
                new_soft_c = soft_c
                new_auxiliary = auxiliary
                break
            # stop if overall auxiliary function has not changed enough
            if (auxiliary-new_auxiliary) <= tol*auxiliary:
                break
        soft_c = new_soft_c
        auxiliary = new_auxiliary

    new_c = new_soft_c.idxmax(axis='columns')
    new_c.name = new_soft_c.columns.name

    timer.pause()

    #assert set(new_c.unique()) == set(centroids.index), "Some clusters are empty"
    return new_c, new_auxiliary, timer.elapsed


# %% Estimate centroids
def estimate_centroids(X: DataFrame, c: Series) -> tuple[DataFrame, int]:
    new_centroids, centroids_time = lloyd.estimate_centroids(X=X, c=c)
    return new_centroids, centroids_time


# %% Compute and show iteration statistics
def _show_iter_stats(
        auxiliary: float | str, utility_loss: float,
        fairness_loss: float, objective: float, i: int,
        n_reassignments: int | str, centroids_time: int | str,
        c_time: int | str, iter_time: int, lgr: Logger
        ) -> dict[str, int | float | str]:
    iter_stats = {
        'auxiliary fn': auxiliary,
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


# %% Run Ziko (2021)'s algorithm
def run(
        X: DataFrame, n_nonsensitive: int, s: Series, n_clusters: int,
        init_centroids: DataFrame, dataset_name: str, init_method: str,
        random_state: int, lambda_: float = LAMBDA_,
        max_iter: int = MAX_ITER, L_: float = L,
        bound_iter: int = BOUND_ITER, tol: float = TOL,
        export: bool = False
        ) -> tuple[Series, DataFrame, DataFrame, DataFrame]:
    assert 0 < n_nonsensitive <= len(X.columns)
    assert s.index.equals(X.index)
    assert n_clusters > 0
    assert set(init_centroids.index) == set(range(n_clusters))
    assert init_centroids.columns.equals(X.columns)
    assert not init_centroids.duplicated().any()
    assert lambda_ >= 0
    assert max_iter >= 1
    assert 0 < L_ <= len(X)
    assert bound_iter >= 1
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

    lgr.info("Initiating Ziko (2021)'s algorithm")

    info = {
        'dataset': [dataset_name],
        'sensitive_attribute': [s.name],
        'n_clusters': [n_clusters],
        'init_method': [init_method],
        'random_state': [random_state],
        'algorithm': [ID],
        'lambda_': [lambda_],
        'max_iter': [max_iter],
        'L': [L_],
        'bound_iter': [bound_iter],
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

    # Not performing L2 normalisation on dataset and initial centroids
    # (as done in the original code) as this will change the dataset
    # and make the outputs not comparable across algorithms

    # run algorithm
    lgr.info("Running algorithm")
    lgr.info("╭────────┬────────────────┬────────────────┬─────────────────┬─────────────┬─────────────────┬──────────────────┬───────────────────┬──────────────────╮")
    lgr.info("│   iter │   auxiliary fn │   utility loss │   fairness loss │   objective │   reassignments │   centroids time │   assignment time │   iteration time │")
    lgr.info("├────────┼────────────────┼────────────────┼─────────────────┼─────────────┼─────────────────┼──────────────────┼───────────────────┼──────────────────┤")

    # initialise centroids
    centroids = init_centroids
    if export:
        centroids.to_csv(os.path.join(centroids_dir, '1.csv'))
    # estimate cluster assignment
    c, c_time = lloyd.estimate_assignment(X=X, centroids=centroids)
    Timer.add_to_all(iter_timer, algo_timer, ns=c_time)
    if export:
        c.to_csv(os.path.join(c_dir, '1.csv'))
    Timer.resume_all(algo_timer, iter_timer)
    # compute objective
    utility_loss = compute_utility_loss(
        X=X, n_nonsensitive=n_nonsensitive, c=c, centroids=centroids
        )
    fairness_loss = compute_fairness_loss(s=s, c=c)
    objective = utility_loss + lambda_*fairness_loss
    print('objective',objective)
    Timer.pause_all(iter_timer, algo_timer)
    # show iteration stats
    stats[1] = _show_iter_stats(
        auxiliary='-', utility_loss=utility_loss,
        fairness_loss=fairness_loss, objective=objective, i=1,
        n_reassignments=len(c), centroids_time='-', c_time=c_time,
        iter_time=iter_timer.elapsed, lgr=lgr
        )
    Timer.reset_all(iter_timer)

    Timer.resume_all(algo_timer, iter_timer)
    converged = False
    auxiliary = '-'
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
        new_c, new_auxiliary, c_time = estimate_assignment(
            X=X, n_nonsensitive=n_nonsensitive, s=s,
            centroids=new_centroids, n_clusters=n_clusters,
            lambda_=lambda_, L_=L_, bound_iter=bound_iter, tol=tol
            )
        Timer.add_to_all(iter_timer, algo_timer, ns=c_time)
        Timer.resume_all(algo_timer, iter_timer)
        # check if overall auxiliary function is diverging
        if auxiliary != '-' and new_auxiliary > auxiliary and i>5:
            # revert to old cluster assignment
            converged = True
            new_c = c
            new_auxiliary = '-'
            n_reassignments = '-'
        else:
            n_reassignments = sum(~new_c.eq(c))
        Timer.pause_all(iter_timer, algo_timer)
        if export:
            new_c.to_csv(os.path.join(c_dir, f'{i}.csv'))
        Timer.resume_all(algo_timer, iter_timer)
        # update clustering state
        centroids = new_centroids
        c = new_c
        auxiliary = new_auxiliary
        old_objective = objective
        # compute objective
        utility_loss = compute_utility_loss(
            X=X, n_nonsensitive=n_nonsensitive, c=c,
            centroids=centroids
            )
        fairness_loss = compute_fairness_loss(s=s, c=c)
        objective = utility_loss + lambda_*fairness_loss
        print('objective',objective)
        Timer.pause_all(iter_timer, algo_timer)
        # show iteration stats
        stats[i] = _show_iter_stats(
            auxiliary=auxiliary, utility_loss=utility_loss,
            fairness_loss=fairness_loss, objective=objective, i=i,
            n_reassignments=n_reassignments,
            centroids_time=centroids_time, c_time=c_time,
            iter_time=iter_timer.elapsed, lgr=lgr
            )
        Timer.reset_all(iter_timer)

        Timer.resume_all(algo_timer)
        # stop if overall auxiliary function is diverging
        if converged:
            Timer.pause_all(algo_timer)
            info['converged'] = True
            info['how_terminated'] = 'overall auxiliary function diverging'
            Timer.resume_all(algo_timer, iter_timer)
            break
        # stop if cluster assignment has not changed
        if n_reassignments == 0 and i>5:
            converged = True
            Timer.pause_all(algo_timer)
            info['converged'] = True
            info['how_terminated'] = 'strict convergence'
            Timer.resume_all(algo_timer, iter_timer)
            break
        # stop if objective is diverging
        if objective > old_objective:
            #Timer.pause_all(algo_timer)
            #info['converged'] = False
            #info['how_terminated'] = 'objective diverging'
            print('i:',1)
            #Timer.resume_all(algo_timer, iter_timer)
            #break
        # stop if objective has not changed enough
        if (old_objective-objective) <= tol*old_objective:
            #Timer.pause_all(algo_timer)
            #info['converged'] = False
            #info['how_terminated'] = 'objective change within tolerance'
            #Timer.resume_all(algo_timer, iter_timer)
            #break
            print('i:',1)

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
        # compute objective
        utility_loss = compute_utility_loss(
            X=X, n_nonsensitive=n_nonsensitive, c=c,
            centroids=centroids
            )
        fairness_loss = compute_fairness_loss(s=s, c=c)
        objective = utility_loss + lambda_*fairness_loss
        Timer.pause_all(iter_timer, algo_timer)
        # show iteration stats
        stats[i] = _show_iter_stats(
            auxiliary='-', utility_loss=utility_loss,
            fairness_loss=fairness_loss, objective=objective, i=i,
            n_reassignments='-', centroids_time=centroids_time,
            c_time='-', iter_time=iter_timer.elapsed, lgr=lgr
            )
        Timer.reset_all(iter_timer)

        info['converged'] = False
        if 'how_terminated' not in info:
            info['how_terminated'] = 'maximum number of iterations'

    lgr.info("╰────────┴────────────────┴────────────────┴─────────────────┴─────────────┴─────────────────┴──────────────────┴───────────────────┴──────────────────╯")
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
    info['utility loss'] = [utility_loss]                  # utility loss after last iteration
    info['fairness loss'] = [fairness_loss]                # fairness loss after last iteration
    info['objective'] = [objective]                        # objective value after last iteration
    info['algo_time_ns'] = [algo_timer.elapsed]            # algorithm running time in nanoseconds
    info = DataFrame.from_dict(info)
    index_col = [
        'dataset', 'sensitive_attribute', 'n_clusters', 'init_method',
        'random_state', 'algorithm', 'lambda_', 'max_iter', 'L',
        'bound_iter', 'tol'
        ]
    info = info.set_index(index_col)
    stats = DataFrame.from_dict(stats, orient='index')
    stats.index.name = 'iter'

    if export:
        info.to_csv(os.path.join(save_dir, 'info.csv'))
        stats.to_csv(os.path.join(save_dir, 'stats.csv'))

    return c, centroids, info, stats


# %% Load algorithm outputs
def load(
        n_clusters: int, dataset_name: str, init_method: str,
        random_state: int, iter_: int = -1
        ) -> tuple[Series, DataFrame, DataFrame, DataFrame]:
    assert n_clusters > 0
    assert iter_ == -1 or iter_ > 0
    load_dir = os.path.join(
        SAVE_DIR, dataset_name, f'k={n_clusters}', init_method,
        f'r={random_state}', ID
        )
    print(f"Loading Ziko (2021)'s clusters and centroids from '{load_dir}'")
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
            'max_iter', 'L', 'bound_iter', 'tol'
            ]
        )
    stats_path = os.path.join(load_dir, 'stats.csv')
    stats = pd.read_csv(stats_path, index_col='iter')
    stats['auxiliary fn'] = stats['auxiliary fn'].transform(
        lambda x: x if x == '-' else float(x)
        )
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


# %% END OF FILE
