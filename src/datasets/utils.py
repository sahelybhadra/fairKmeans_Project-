#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Common dataset utilities.

Attributes
----------
HOT : float
    Value for hot in one hot encoding.

Routine Listings
----------------
count_nonsensitive()
    Get number of non-sensitive attributes in the dataset.
describe_ser()
    Generate descriptive statistics of a series.
download_file_from_url()
    Download a file from a URL.
one_hot_encode()
    One hot encode a categorical attribute.
standardise()
    Standardise a continuous attribute.

"""


# %% Libraries
import math
import numpy as np
import os
import pandas as pd
import requests
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler


# %% Constants
# Value for hot in one hot encoding
HOT: float = 1 / math.sqrt(2)    # so that sum of squared differences is 0 or 1


# %% Get number of non-sensitive attributes in the dataset
def count_nonsensitive(X: DataFrame) -> int:
    r"""
    Get number of non-sensitive attributes in the dataset.

    Parameters
    ----------
    X : DataFrame
        Dataset.

    Returns
    -------
    int
        Number of non-sensitive attributes.

    """
    return X.columns.map(lambda x: x.split(':')[0]).nunique()


# %% Generate descriptive statistics of a series
def describe_ser(ser: Series) -> Series:
    r"""
    Generate descriptive statistics of a series.

    Descriptive statistics include the minimum, maximum, mean, standard
    deviation, and percentage of missing values.

    Parameters
    ----------
    ser : Series
        Series.

    Raises
    ------
    TypeError
        If the series is not numerical.

    Returns
    -------
    Series
        Descriptive statistics of the series. Has the following index:

        +-----------+
        | min       |
        +-----------+
        | max       |
        +-----------+
        | mean      |
        +-----------+
        | std       |
        +-----------+
        | % missing |
        +-----------+

    """
    if not np.issubdtype(ser, np.number):
        raise TypeError
    stats = dict()
    stats['min'] = ser.min()
    stats['max'] = ser.max()
    stats['mean'] = ser.mean()
    stats['std'] = ser.std()
    stats['% missing'] = 100 * sum(ser.isna()) / len(ser)
    return Series(stats, dtype=object, name=ser.name)


# %% Download a file from a URL
def download_file_from_url(file_url: str, save_dir: str) -> str:
    r"""
    Download a file from a URL.

    Parameters
    ----------
    file_url : str
        URL of file to be downloaded.
    save_dir : str
        Directory where the downloaded file is to be saved.

    Returns
    -------
    str
        Path of the downloaded file.

    """
    chunk_size = 32768
    print(f"Downloading from '{file_url}'")
    filename = file_url.split('/')[-1]
    filepath = os.path.join(save_dir, filename)
    with requests.get(file_url, stream=True) as response:
        os.makedirs(save_dir, exist_ok=True)
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size):
                if chunk:    # filter out keep-alive new chunks
                    file.write(chunk)
    print(f"Downloaded to '{filepath}'")
    return filepath


# %% One hot encode a categorical attribute
def one_hot_encode(ser: Series) -> DataFrame:
    r"""
    One hot encode a categorical attribute.

    The new one hot attributes have names that follow the pattern
    '``attribute:category``' where ``category`` is an attribute
    category. The default value for hot is :math:`\frac{1}{\sqrt{2}}`.
    This ensures that the maximum squared distance between any two
    objects for the replaced categorical attribute is 1.

    Parameters
    ----------
    ser : Series
        Series corresponding to the attribute.

    Returns
    -------
    DataFrame
        One hot encoded attribute.

    """
    attr = ser.name
    oha_sers = list()
    for cat in sorted(ser.unique()):
        oha_ser = ser.apply(lambda x: HOT if x == cat else 0)
        oha_ser.name = f'{attr}:{cat}'
        oha_sers.append(oha_ser)
    oha_frame = pd.concat(oha_sers, axis='columns')
    assert all(oha_frame.sum(axis='columns') == HOT)
    return oha_frame


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


# %% END OF FILE
