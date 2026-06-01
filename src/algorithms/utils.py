#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Common algorithm utilities.

Routine Listings
----------------
compute_centroid_shift()
    Compute the centroid shift.
compute_distances()
    Compute objects' squared Euclidean distances from all centroids.
generate_windows()
    Generate pooling windows.
get_last_iteration()
    Get last iteration of the algorithm.
get_object_losses()
    Get objects' losses.
print_trace()
    Print algorithm trace.
standardise()
    Standardise a continuous attribute.
sumpool()
    Apply sum pooling over a histogram.

"""


# %% Libraries
import numpy as np
import os
import pandas as pd
import re
import zipfile
from collections.abc import Mapping
from pandas import DataFrame, Index, IntervalIndex, Series
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate


# %% Compute the centroid shift
def compute_centroid_shift(
        centroids: DataFrame, new_centroids: DataFrame
        ) -> float:
    r"""
    Compute the centroid shift.

    Quantified as the Frobenius norm of the difference in centroids.

    Parameters
    ----------
    centroids : DataFrame
        Current centroids.
    new_centroids : DataFrame
        New centroids.

    Returns
    -------
    float
        Frobenius norm of the difference in centroids.

    """
    return np.linalg.norm(new_centroids.subtract(centroids), ord='fro')


# %% Compute objects' squared Euclidean distances from all centroids
def compute_distances(X: DataFrame, centroids: DataFrame) -> DataFrame:
    r"""
    Compute objects' squared Euclidean distances from all centroids.

    Parameters
    ----------
    X : DataFrame
        Dataset.
    centroids : DataFrame
        Centroids.

    Returns
    -------
    DataFrame
        Objects' distances from all centroids.

    """
    distances = cdist(X, centroids, metric='sqeuclidean')
    return DataFrame(distances, index=X.index, columns=centroids.index)


# %% Generate pooling windows
def generate_windows(index: Index, window_size: int) -> IntervalIndex:
    r"""
    Generate pooling windows.

    The first pooling window is such that its rightmost element is the
    first element in `index`, and the last pooling window is such that
    its leftmost element is the last element in `index`.

    Parameters
    ----------
    index : Index
        Index over which the windows are to be generated.
    window_size : int
        Pooling window size.

    Returns
    -------
    IntervalIndex
        Pooling windows.

    """
    index = index.unique()
    assert window_size <= len(index)
    index = index.sort_values()
    last_idx = len(index) - 1
    offset = window_size - 1
    windows = [
        (index[max(0, left)], index[min(last_idx, left+offset)])
        for left in range(-offset, len(index))
        ]
    windows = IntervalIndex.from_tuples(
        windows, closed='both', name=index.name
        )
    return windows


# %% Get last iteration of the algorithm
def get_last_iteration(save_dir: str) -> int:
    r"""
    Get last iteration of the algorithm.

    Parameters
    ----------
    save_dir : str
        Directory where the algorithm outputs are saved.

    Raises
    ------
    ValueError
        If the last iteration differs for clusters and centroids.

    Returns
    -------
    int
        Last iteration.

    """
    last = dict()
    for output in ['clusters', 'centroids']:
        dirpath = os.path.join(save_dir, output)
        with zipfile.ZipFile(f'{dirpath}.zip') as zf:
            paths = zf.namelist()
        for path in paths:
            filename = path.split('/')[-1]
            match = re.fullmatch(r"^(\d+)\.csv$", filename)
            if match is None:
                raise ValueError(fr"File name in '{path}' does not match the pattern '^(\d+)\.csv$'")
            i = int(match.group(1))
            if output not in last or last[output] < i:
                last[output] = i
    if last['clusters'] != last['centroids']:
        raise ValueError(f"Last iteration differs for clusters ({last['clusters']}) and centroids ({last['centroids']})")
    return last['clusters']


# %% Get objects' losses
def get_object_losses(distances: DataFrame, c: Series) -> Series:
    r"""
    Get objects' losses.

    Parameters
    ----------
    distances : DataFrame
        Objects' distances from all centroids.
    c : Series
        Cluster assignment.

    Returns
    -------
    Series
        Objects' losses.

    """
    idx, cols = pd.factorize(c)
    distances = distances.reindex(cols, axis='columns')
    object_losses = distances.to_numpy()[np.arange(len(c)), idx]
    return Series(object_losses, index=c.index, name='loss')


# %% Print algorithm trace
def print_trace(
        info: DataFrame, stats: DataFrame,
        stats_formatters: Mapping[str, str]
        ):
    r"""
    Print algorithm trace from the passed parameters.

    Parameters
    ----------
    info : DataFrame
        Information summary of the run.
    stats : DataFrame
        Statistics of each iteration in the run.
    stats_formatters : Mapping[str, str]
        Formatters for printing iteration statistics.

    Returns
    -------
    None.

    """
    assert len(info) == 1
    assert info.iloc[0]['n_iter'] == len(stats)
    assert info.iloc[0]['objective'] == stats.iloc[-1]['objective']
    info_str = tabulate(
        info.index.to_frame(index=False),
        headers='keys',
        tablefmt='rounded_outline',
        stralign='right',
        showindex=False
        )
    print(info_str)
    stats_str = tabulate(
        stats.transform(stats_formatters),
        headers='keys',
        tablefmt='rounded_outline',
        stralign='right',
        disable_numparse=True
        )
    print(stats_str)
    converged, how_terminated, n_iter, algo_time_ns = \
        info.iloc[0][['converged', 'how_terminated', 'n_iter', 'algo_time_ns']]
    if converged:
        print(f"Converged at iteration {n_iter}: {how_terminated}")
    else:
        print(f"Terminated at iteration {n_iter}: {how_terminated}")
    print(f"Algorithm running time: {algo_time_ns * 1e-9} seconds")


# %% Standardise a continuous attribute
def standardise(ser: Series) -> Series:
    r"""
    Standardise a continuous attribute.

    The continuous attribute is standardised, *i.e.*, zero mean and
    unit variance.

    Parameters
    ----------
    ser : Series
        Series corresponding to the attribute.

    Returns
    -------
    Series
        Standardised attribute.

    """
    values = ser.values.reshape(-1, 1)
    standardised = StandardScaler().fit_transform(values)
    return Series(standardised.reshape(1, -1)[0], index=ser.index)


# %% Apply sum pooling over a histogram
def sumpool(hist: Series, windows: IntervalIndex) -> Series:
    r"""
    Apply sum pooling over a histogram.

    Parameters
    ----------
    hist : Series
        Histogram.
    windows : IntervalIndex
        Pooling windows.

    Returns
    -------
    Series
        Sum pooled histogram.

    """
    pooled = {
        w: hist[(hist.index >= w.left) & (hist.index <= w.right)].sum()
        for w in windows
        }
    pooled = Series(pooled, name=hist.name)
    pooled.index.name = windows.name
    return pooled


# %% END OF FILE
