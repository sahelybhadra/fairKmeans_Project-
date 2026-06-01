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
from numpy.typing import NDArray
from pandas import DataFrame, IntervalIndex, Series
from tabulate import tabulate

from . import lloyd, utils, SAVE_DIR
from .. import logger
from ..timer import Timer


# %% Algorithm identifier
ID: str = 'fairkmeans_prc'


# %% Parameters
LAMBDA_: float = .5     # trade-off between fairness (1) and utility (0)
WINDOW_SIZE: int = 3    # pooling window size
MAX_ITER: int = 100     # maximum number of iterations


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


# %% Estimate h_X
def estimate_pooled_dataset_pmf(s: Series, window_size: int) -> Series:
    counts = s.value_counts(sort=False).sort_index()
    windows = utils.generate_windows(
        index=counts.index, window_size=window_size
        )
    pooled_counts = utils.sumpool(hist=counts, windows=windows)
    return pooled_counts.divide(pooled_counts.sum())


# %% Get pooled counts of sensitive attribute's values in each cluster
def get_pooled_cluster_counts(
        s: Series, c: Series, windows: IntervalIndex
        ) -> DataFrame:
    # get counts of sensitive attribute's values in each cluster
    counts = c.groupby(s, sort=False).value_counts(sort=False)
    counts = counts.unstack(level=c.name, fill_value=0)
    counts = counts.sort_index(axis='index').sort_index(axis='columns')
    # get pooled counts of sensitive attribute's values in each cluster
    return counts.apply(utils.sumpool, windows=windows)


# %% Compute utility loss
def compute_utility_loss(
        X: DataFrame, n_nonsensitive: int, c: Series,
        centroids: DataFrame
        ) -> float:
    return lloyd.compute_objective(
        X=X, n_nonsensitive=n_nonsensitive, c=c, centroids=centroids
        )


# %% Compute fairness loss
def compute_fairness_loss(s: Series, c: Series, window_size: int) -> float:
    # estimate h for the dataset
    h_X = estimate_pooled_dataset_pmf(s=s, window_size=window_size)
    # get pooled counts of sensitive attribute's values in each cluster
    pooled_counts = get_pooled_cluster_counts(s=s, c=c, windows=h_X.index)
    # get pooled size of each cluster
    pooled_sizes = pooled_counts.sum(axis='index')
    # estimate h for all clusters
    h = pooled_counts.divide(pooled_sizes, axis='columns')
    # compute \sum_{C} \sum_{w} |h_C(w) - h_X(w)|
    sum_losses = np.nansum(h.subtract(h_X, axis='index').abs())
    return sum_losses / len(h.columns)


# %% Compute change in utility loss on reassigning an object
def _compute_diff_utility_loss(
        C: int, d: Series, n_objects: int, n_nonsensitive: int
        ) -> Series:
    return (d - d.loc[C]) / (n_objects * n_nonsensitive)


# %% Compute change in fairness loss on reassigning an object
def _compute_diff_fairness_loss(
        S_windows: NDArray, C: int, pooled_counts: DataFrame,
        pooled_sizes: Series, h_X: Series, n_clusters: int
        ) -> Series:
    # estimate h for all current clusters
    h = pooled_counts.divide(pooled_sizes, axis='columns')
    # compute fairness loss due to each cluster
    c_losses = h.subtract(h_X, axis='index').abs().sum(axis='index')
    # update clustering to reflect `x` being removed from `C`
    n_changes = S_windows.sum()
    new_pooled_counts = pooled_counts.copy()
    new_pooled_counts.loc[S_windows, C] -= 1
    new_pooled_sizes = pooled_sizes.copy()
    new_pooled_sizes.loc[C] -= n_changes
    # estimate h for `C` with `x` removed
    new_h_C = new_pooled_counts.loc[:, C].divide(new_pooled_sizes.loc[C])
    # compute fairness loss for `C` with `x` removed
    new_C_loss = new_h_C.subtract(h_X).abs().sum()
    # update clustering to reflect `x` being added to each cluster
    new_pooled_counts.loc[S_windows] += 1
    new_pooled_sizes += n_changes
    # estimate h for each cluster with `x` added
    new_h = new_pooled_counts.divide(new_pooled_sizes, axis='columns')
    # compute fairness loss for each cluster with `x` added
    new_c_losses = new_h.subtract(h_X, axis='index').abs().sum(axis='index')
    # compute change in fairness loss of `C` due `x` being removed
    diff_C = new_C_loss - c_losses.loc[C]
    # compute change in fairness losses of each cluster due to `x` being added
    diff_new_c = new_c_losses.subtract(c_losses)
    # compute change in fairness loss
    diff_fairness_loss = diff_new_c + diff_C
    diff_fairness_loss.loc[C] = 0
    diff_fairness_loss = diff_fairness_loss / n_clusters
    return diff_fairness_loss


# %% Compute change in objective on reassigning an object
def _compute_diffs(
        x: int, X: DataFrame, n_nonsensitive: int, s: Series,
        c: Series, centroids: DataFrame, h_X: Series, n_clusters: int,
        lambda_: float
        ) -> tuple[Series, Series, Series]:
    C = c.loc[x]
    utility_loss = dict()
    fairness_loss = dict()
    objective = dict()
    for new_C in range(n_clusters):
        c.loc[x] = new_C
        utility_loss[new_C] = compute_utility_loss(
            X=X, n_nonsensitive=n_nonsensitive, c=c, centroids=centroids
            )
        fairness_loss[new_C] = compute_fairness_loss(s=s, c=c, h_X=h_X)
        objective[new_C] = \
            (1-lambda_)*utility_loss[new_C] + lambda_*fairness_loss[new_C]
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
        centroids: DataFrame, n_clusters: int, window_size: int,
        lambda_: float
        ) -> tuple[Series, int]:
    assert 0 < n_nonsensitive <= len(X.columns)
    assert s.index.equals(X.index)
    assert c.index.equals(X.index)
    assert n_clusters > 0
    assert set(c.unique()) == set(centroids.index) == set(range(n_clusters))
    assert centroids.columns.equals(X.columns)
    assert not centroids.duplicated().any()
    assert window_size >= 1
    assert 0 <= lambda_ <= 1

    timer = Timer()
    timer.resume()

    distances = utils.compute_distances(X=X, centroids=centroids)
    n_objects = len(X)
    # estimate h for the dataset
    h_X = estimate_pooled_dataset_pmf(s=s, window_size=window_size)
    # get pooled counts of sensitive attribute's values in each cluster
    pooled_counts = get_pooled_cluster_counts(s=s, c=c, windows=h_X.index)
    # get pooled size of each cluster
    pooled_sizes = pooled_counts.sum(axis='index')
    new_c = c.copy()

    # Reorder objects in `distances` so that the ones with least
    # maximum distances are seen first. This way, the objects with
    # higher maximum distances would likely remain in their nearest
    # clusters.
    reordered_index = distances.max(axis='columns').sort_values().index
    reordered_distances = distances.reindex(index=reordered_index)

    # reassign objects to clusters such that the objective decreases
    for x, d in reordered_distances.iterrows():
        S_windows = h_X.index.contains(s.loc[x])
        C = new_c.loc[x]
        # compute change in utility loss on reassigning `x`
        diff_utility_loss = _compute_diff_utility_loss(
            C=C, d=d, n_objects=n_objects,
            n_nonsensitive=n_nonsensitive
            )
        # compute change in fairness loss on reassigning `x`
        diff_fairness_loss = _compute_diff_fairness_loss(
            S_windows=S_windows, C=C, pooled_counts=pooled_counts,
            pooled_sizes=pooled_sizes, h_X=h_X, n_clusters=n_clusters
            )
        # compute resulting change in objective on reassigning `x`
        diff_objective = (1-lambda_) * diff_utility_loss
        diff_objective = diff_objective.add(lambda_ * diff_fairness_loss)

        # # Uncomment this block only if you want to check the correctness
        # timer.pause()
        # u2, f2, o2 = _compute_diffs(
        #     x=x, X=X, n_nonsensitive=n_nonsensitive, s=s, c=new_c,
        #     centroids=centroids, h_X=h_X, n_clusters=n_clusters,
        #     lambda_=lambda_
        #     )
        # assert diff_utility_loss.subtract(u2).abs().sum() < 1e-15
        # assert diff_fairness_loss.subtract(f2).abs().sum() < 1e-15
        # assert diff_objective.subtract(o2).abs().sum() < 1e-15
        # timer.resume()
        # # Block ends here

        # best cluster is one where the difference in the new objective
        # and old objective is the largest negative number
        if diff_objective.min()<-0.00001:
            best_C = diff_objective.idxmin()
        else:
            #print('difference is less than tolarance')
            best_C=C
        if best_C != C:
            new_c.loc[x] = best_C
            pooled_counts.loc[S_windows, C] -= 1
            pooled_counts.loc[S_windows, best_C] += 1
            pooled_sizes.loc[C] -= S_windows.sum()
            pooled_sizes.loc[best_C] += S_windows.sum()
            timer.pause()
            assert set(new_c.unique()) == set(centroids.index), "Some clusters are empty"
            timer.resume()

    timer.pause()

    assert set(new_c.unique()) == set(centroids.index), "Some clusters are empty"
    return new_c, timer.elapsed


# %% Estimate centroids
def estimate_centroids(X: DataFrame, c: Series) -> tuple[DataFrame, int]:
    new_centroids, centroids_time = lloyd.estimate_centroids(X=X, c=c)
    return new_centroids, centroids_time


# %% Compute and show iteration statistics
def _show_iter_stats(
        X: DataFrame, n_nonsensitive: int, s: Series, c: Series,
        centroids: DataFrame, window_size: int, lambda_: float, i: int,
        n_reassignments: int | str, centroids_time: int | str,
        c_time: int | str, iter_time: int, lgr: Logger
        ) -> dict[str, int | float | str]:
    utility_loss = compute_utility_loss(
        X=X, n_nonsensitive=n_nonsensitive, c=c, centroids=centroids
        )
    fairness_loss = compute_fairness_loss(s=s, c=c, window_size=window_size)
    objective = (1-lambda_)*utility_loss + lambda_*fairness_loss
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


# %% Run the algorithm
def run(
        X: DataFrame, n_nonsensitive: int, s: Series, n_clusters: int,
        init_centroids: DataFrame, dataset_name: str, init_method: str,
        random_state: int, window_size: int = WINDOW_SIZE,
        lambda_: float = LAMBDA_, max_iter: int = MAX_ITER,
        export: bool = False
        ) -> tuple[Series, DataFrame, DataFrame, DataFrame]:
    assert 0 < n_nonsensitive <= len(X.columns)
    assert s.index.equals(X.index)
    assert n_clusters > 0
    assert set(init_centroids.index) == set(range(n_clusters))
    assert init_centroids.columns.equals(X.columns)
    assert not init_centroids.duplicated().any()
    assert window_size >= 1
    assert 0 <= lambda_ <= 1
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

    lgr.info("Initiating fair k-means with proportional representation"
             " along a continuous sensitive attribute")

    info = {
        'dataset': [dataset_name],
        'sensitive_attribute': [s.name],
        'n_clusters': [n_clusters],
        'init_method': [init_method],
        'random_state': [random_state],
        'algorithm': [ID],
        'lambda_': [lambda_],
        'window_size': [window_size],
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
        centroids=centroids, window_size=window_size, lambda_=lambda_,
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
            window_size=window_size, lambda_=lambda_
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
            centroids=centroids, window_size=window_size,
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
            centroids=centroids, window_size=window_size,
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
        'random_state', 'algorithm', 'lambda_', 'window_size',
        'max_iter'
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
    print(f"Loading clusters and centroids generated by fair k-means"
          f" with proportional representation along a continuous"
          f" sensitive attribute from '{load_dir}'")
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
            'window_size', 'max_iter'
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


# %% END OF FILE
