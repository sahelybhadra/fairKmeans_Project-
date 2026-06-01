#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
"""


# %% Libraries
import numpy as np
import pandas as pd
from importlib import import_module
from numpy.typing import NDArray
from pandas import DataFrame, IntervalIndex, Series
from sklearn.cluster import _kmeans
from tabulate import tabulate

from src import init, metrics
from src.algorithms import lloyd, fairkmeans_prc, utils
from src.timer import Timer


# %% Parameters
DATASET_NAME = 'crime'           # ['adult', 'compas', 'crime', 'motor']
N_CLUSTERS = 4
RANDOM_STATE = 0
INIT_METHOD = 'kmeans_plusplus'

LAMBDA_: float = .5
MAX_ITER: int = 10
WINDOW_SIZE: float = 3
TOL: float = 1e-4


# %% Load dataset
dataset_module = import_module(f"src.datasets.{DATASET_NAME}")
_, X, s, n_nonsensitive = dataset_module.load()


# %% Load initialised centroid
init_centroids = init.load(
    dataset_name=DATASET_NAME, n_clusters=N_CLUSTERS,
    init_method=INIT_METHOD, random_state=RANDOM_STATE
    )


# %% Run our algorithm
n_clusters = N_CLUSTERS
dataset_name = DATASET_NAME
init_method = INIT_METHOD
random_state = RANDOM_STATE
max_iter = MAX_ITER
window_size = WINDOW_SIZE
tol = TOL
lambda_ = LAMBDA_

algo_timer = Timer()
iter_timer = Timer()

Timer.resume_all(algo_timer)
# get counts of sensitive attribute's values in the dataset
dataset_count = s.value_counts(sort=False).sort_index()
# generated pooling window instances
windows = utils.generate_windows(
    index=dataset_count.index, window_size=window_size
    )
# get pooled counts of sensitive attribute's values in the dataset
dataset_pooled_count = utils.sumpool(hist=dataset_count, windows=windows)
# compute h(X,·)
dataset_pooled_prop = dataset_pooled_count.divide(dataset_pooled_count.sum())
# set tolerance dependent on the dataset (as done in sklearn's kmeans)
tol_ = _kmeans._tolerance(X, tol)
Timer.pause_all(algo_timer)

# run algorithm
print("Running algorithm")
centroids_timer = Timer()
Timer.resume_all(algo_timer, iter_timer, centroids_timer)
# initialise centroids
centroids = init_centroids
Timer.pause_all(centroids_timer, iter_timer, algo_timer)
centroids_time = centroids_timer.elapsed
# estimate cluster assignment
c, c_time = lloyd.estimate_assignment(X=X, centroids=centroids)
Timer.add_to_all(iter_timer, algo_timer, ns=c_time)

# compute stats
print("╭────────┬────────────────┬─────────────────┬─────────────┬─────────────────┬──────────────────┬───────────────────┬──────────────────╮")
print("│   iter │   utility loss │   fairness loss │   objective │   reassignments │   centroids time │   assignment time │   iteration time │")
print("├────────┼────────────────┼─────────────────┼─────────────┼─────────────────┼──────────────────┼───────────────────┼──────────────────┤")
utility_loss = fairkmeans_prc.compute_utility_loss(X=X, n_nonsensitive=n_nonsensitive, c=c, centroids=centroids)
fairness_loss = fairkmeans_prc.compute_fairness_loss(s=s, c=c, dataset_pooled_prop=dataset_pooled_prop)
objective = (1-lambda_)*utility_loss + lambda_*fairness_loss
print(f"│ {1:6} │ {utility_loss:>14.5e} │ {fairness_loss:>15.5e} │ {objective:>11.5e} │ {len(c):>15} │ {centroids_time:>16} │ {c_time:>17} │ {iter_timer.elapsed:>16} │")
Timer.reset_all(iter_timer)

Timer.resume_all(algo_timer, iter_timer)
converged = False
for i in range(2, MAX_ITER):
    Timer.pause_all(iter_timer, algo_timer)
    # estimate centroids
    new_centroids, centroids_time = fairkmeans_prc.estimate_centroids(X=X, c=c)
    assert new_centroids.index.equals(centroids.index)
    assert new_centroids.columns.equals(centroids.columns)
    Timer.add_to_all(
        iter_timer, algo_timer, ns=centroids_time
        )
    # estimate cluster assignment
    new_c, c_time = fairkmeans_prc.estimate_assignment(
        X=X, n_nonsensitive=n_nonsensitive, s=s, c=c,
        centroids=new_centroids, n_clusters=n_clusters,
        dataset_pooled_prop=dataset_pooled_prop, lambda_=lambda_
        )
    Timer.add_to_all(iter_timer, algo_timer, ns=c_time)

    Timer.resume_all(algo_timer, iter_timer)
    n_reassignments = sum(~new_c.eq(c))
    c = new_c
    centroids = new_centroids
    Timer.pause_all(iter_timer, algo_timer)

    # compute stats
    utility_loss = fairkmeans_prc.compute_utility_loss(X=X, n_nonsensitive=n_nonsensitive, c=c, centroids=centroids)
    fairness_loss = fairkmeans_prc.compute_fairness_loss(s=s, c=c, dataset_pooled_prop=dataset_pooled_prop)
    objective = (1-lambda_)*utility_loss + lambda_*fairness_loss
    print(f"│ {i:6} │ {utility_loss:>14.5e} │ {fairness_loss:>15.5e} │ {objective:>11.5e} │ {n_reassignments:>15} │ {centroids_time:>16} │ {c_time:>17} │ {iter_timer.elapsed:>16} │")
    Timer.reset_all(iter_timer)

    Timer.resume_all(algo_timer)
    # stop if cluster assignment has not changed
    if n_reassignments == 0:
        converged = True
        Timer.pause_all(algo_timer)
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
    new_centroids, centroids_time = fairkmeans_prc.estimate_centroids(X=X, c=c)
    assert new_centroids.index.equals(centroids.index)
    assert new_centroids.columns.equals(centroids.columns)
    Timer.add_to_all(
        iter_timer, algo_timer, ns=centroids_time
        )

    Timer.resume_all(algo_timer, iter_timer)
    centroids = new_centroids
    Timer.pause_all(iter_timer, algo_timer)

    # compute stats
    utility_loss = fairkmeans_prc.compute_utility_loss(X=X, n_nonsensitive=n_nonsensitive, c=c, centroids=centroids)
    fairness_loss = fairkmeans_prc.compute_fairness_loss(s=s, c=c, dataset_pooled_prop=dataset_pooled_prop)
    objective = (1-lambda_)*utility_loss + lambda_*fairness_loss
    print(f"│ {i:6} │ {utility_loss:>14.5e} │ {fairness_loss:>15.5e} │ {objective:>11.5e} │ {'-':>15} │ {centroids_time:>16} │ {'-':>17} │ {iter_timer.elapsed:>16} │")
    Timer.reset_all(iter_timer)

print("╰────────┴────────────────┴─────────────────┴─────────────┴─────────────────┴──────────────────┴───────────────────┴──────────────────╯")


# %% Evaluate
# scores = metrics.evaluate(X=X, n_nonsensitive=n_nonsensitive, s=s, c=c, centroids=centroids, window_size=3)
# print(tabulate(scores.to_frame(), tablefmt='rounded_outline'))


# %% END OF FILE
