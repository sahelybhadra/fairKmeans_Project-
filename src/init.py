#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Utilities for initialising centroids.

Attributes
----------
INIT_DIR : str
    Subdirectory that would contain the initialised centroids.

Routine Listings
----------------
random()
    Sample initial centroids from the dataset.
kmeans_plusplus()
    Initialise centroids using :math:`k`-means++ [1]_.
export()
    Export initialised centroids.
load()
    Load initialised centroids.

References
----------
.. [1] David Arthur, Sergei Vassilvitskii. `k-means++: The Advantages
   of Careful Seeding
   <http://dl.acm.org/citation.cfm?id=1283383.1283494>`_. SODA 2007.

"""


# %% Libraries
import os
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import kmeans_plusplus as sklearn_kmeans_plusplus


# %% Paths
# Subdirectory that would contain the initialised centroids
INIT_DIR: str = 'data/init'


# %% Sample initial centroids from the dataset
def random(X: DataFrame, n_clusters: int, random_state: int) -> DataFrame:
    r"""
    Sample initial centroids from the dataset.

    Parameters
    ----------
    X : DataFrame
        Dataset.
    n_clusters : int
        Number of clusters.
    random_state : int
        Random state.

    Returns
    -------
    DataFrame
        Initialised centroids.

    """
    X = X.drop_duplicates()
    assert len(X) >= n_clusters > 0
    print("Sampling initial centroids from the dataset")
    centroids = X.sample(n=n_clusters, random_state=random_state)
    centroids = centroids.reset_index(drop=True)
    centroids.index.name = 'cluster'
    print(f"  shape: {centroids.shape}")
    return centroids


# %% Initialise centroids using k-means++
def kmeans_plusplus(
        X: DataFrame, n_clusters: int, random_state: int
        ) -> DataFrame:
    r"""
    Initialise centroids using :math:`k`-means++ [1]_.

    Uses scikit-learn's implementation of :math:`k`-means++ [2]_.

    Parameters
    ----------
    X : DataFrame
        Dataset.
    n_clusters : int
        Number of clusters.
    random_state : int
        Random state.

    Returns
    -------
    DataFrame
        Initialised centroids.

    References
    ----------
    .. [1] David Arthur, Sergei Vassilvitskii. `k-means++: The
       Advantages of Careful Seeding
       <http://dl.acm.org/citation.cfm?id=1283383.1283494>`_. SODA
       2007.
    .. [2] https://scikit-learn.org/stable/modules/generated/sklearn.cluster.kmeans_plusplus.html

    """
    assert len(X.drop_duplicates()) >= n_clusters > 0
    print("Initialising centroids using k-means++")
    centroids, indices = sklearn_kmeans_plusplus(
        X=X.to_numpy(), n_clusters=n_clusters, random_state=random_state
        )
    assert (centroids == X.iloc[indices].to_numpy()).all(axis=None)
    centroids = DataFrame(centroids, columns=X.columns)
    centroids.index.name = 'cluster'
    print(f"  shape: {centroids.shape}")
    return centroids


# %% Export initialised centroids
def export(
        centroids: DataFrame, dataset_name: str, init_method: str,
        random_state: int
        ):
    r"""
    Export initialised centroids.

    The centroids are exported to
    '`INIT_DIR`/`dataset_name`/k=`n_clusters`/`init_method`' having
    file name
    '`dataset_name`.k`n_clusters`.`init_method`.r`random_state`.csv'.

    Parameters
    ----------
    centroids : DataFrame
        Initialised centroids.
    dataset_name : str
        Name of the dataset.
    init_method : str
        Method for initialisation.
    random_state : int
        Random state.

    Returns
    -------
    None.

    """
    n_clusters = len(centroids)
    save_dir = os.path.join(
        INIT_DIR, dataset_name, f'k={n_clusters}', f'{init_method}'
        )
    filename = f'{dataset_name}.k{n_clusters}.{init_method}.r{random_state}.csv'
    filepath = os.path.join(save_dir, filename)
    print(f"Exporting centroids to '{filepath}'")
    os.makedirs(save_dir, exist_ok=True)
    centroids.to_csv(filepath, mode='x')


# %% Load initialised centroids
def load(
        dataset_name: str, n_clusters: int, init_method: str,
        random_state: int
        ) -> DataFrame:
    r"""
    Load initialised centroids.

    Assumes that the centroids are saved in
    '`INIT_DIR`/`dataset_name`/k=`n_clusters`/`init_method`' having
    file name
    '`dataset_name`.k`n_clusters`.`init_method`.r`random_state`.csv'.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    n_clusters : int
        Number of clusters.
    init_method : str
        Method for initialisation.
    random_state : int
        Random state.

    Returns
    -------
    DataFrame
        Initialised centroids.

    """
    load_dir = os.path.join(
        INIT_DIR, dataset_name, f'k={n_clusters}', f'{init_method}'
        )
    filename = f'{dataset_name}.k{n_clusters}.{init_method}.r{random_state}.csv'
    filepath = os.path.join(load_dir, filename)
    print(f"Loading centroids from '{filepath}'")
    centroids = pd.read_csv(filepath, index_col=0)
    assert len(centroids) == n_clusters
    centroids.columns.name = 'attribute'
    print(f"  shape: {centroids.shape}")
    return centroids


# %% END OF FILE
