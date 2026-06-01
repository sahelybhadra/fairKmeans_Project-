#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Utilities for processing the Adult dataset [1]_.

The attributes for the original dataset are taken from its
'adult.names' file. For the processed version, the original dataset is
cleaned as per Le Quy (2022) [2]_. The non-sensitive attributes are
selected following Le Quy (2022) [2]_, and sensitive attribute ('age')
following Jiang (2022) [3]_.

Attributes
----------
DATASET_DIR : str
    Subdirectory that would contain the original and processed
    datasets.
DATASET_NAME : str
    Name of the processed dataset.
ATTRIBUTES : list[str]
    Attributes in the original dataset.
CATEGORICAL : list[str]
    Categorical attributes.
NONSENSITIVE : list[str]
    Non-sensitive attributes used in Le Quy (2022) [2]_.
SENSITIVE : str
    Sensitive attribute used in Jiang (2022) [3]_.

Routine Listings
----------------
download()
    Download the original Adult dataset.
load_original()
    Load the original Adult dataset.
process()
    Process the original Adult dataset.
load()
    Load the processed Adult dataset.

References
----------
.. [1] Barry Becker, Ronny Kohavi. `Adult
   <https://doi.org/10.24432/C5XW20>`_. UCI Machine Learning
   Repository. 1996.
.. [2] Tai Le Quy, Arjun Roy, Vasileios Iosifidis, Wenbin Zhang, Eirini
   Ntoutsi. `A survey on datasets for fairness-aware machine learning
   <https://doi.org/10.1002/widm.1452>`_. WIREs Data Mining and
   Knowledge Discovery. 2022.
.. [3] Zhimeng Jiang, Xiaotian Han, Chao Fan, Fan Yang, Ali Mostafavi,
   Xia Hu. `Generalized Demographic Parity for Group Fairness
   <https://openreview.net/forum?id=YigKlMJwjye>`_. ICLR 2022.

"""


# %% Libraries
import os
import pandas as pd
import zipfile
from pandas import DataFrame, Series
from tabulate import tabulate

from . import utils


# %% Paths
# Subdirectory that would contain the Adult dataset
DATASET_DIR: str = 'data/datasets/adult'


# %% Dataset name
DATASET_NAME: str = 'adult'


# %% Attributes taken from 'adult.names'
ATTRIBUTES: list[str] = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'annual-income'
    ]


# %% Categorical attributes taken from 'adult.names'
CATEGORICAL: list[str] = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country', 'annual-income'
    ]


# %% Non-sensitive attributes used in Le Quy (2022)
NONSENSITIVE: list[str] = [
    'workclass', 'education-num', 'occupation', 'capital-gain',
    'capital-loss', 'hours-per-week',
    ]


# %% Sensitive attribute used in Jiang (2022)
SENSITIVE: str = 'age'


# %% Download the original Adult dataset
def download(
        url: str = 'https://archive.ics.uci.edu/static/public/2/adult.zip',
        overwrite: bool = False
        ):
    r"""
    Download the original Adult dataset [1]_.

    Assumes that the data file name is 'adult.zip'. Saves the
    downloaded dataset in '`DATASET_DIR`/_original'.

    Parameters
    ----------
    url : str, optional
        URL of the dataset. The default is
        'https://archive.ics.uci.edu/static/public/2/adult.zip'.
    overwrite : bool, optional
        Whether to overwrite existing files. The default is False.

    Returns
    -------
    None.

    References
    ----------
    .. [1] Barry Becker, Ronny Kohavi. `Adult
       <https://doi.org/10.24432/C5XW20>`_. UCI Machine Learning
       Repository. 1996.

    """
    print("Preparing to download the original Adult dataset")
    save_dir = os.path.join(DATASET_DIR, '_original')
    # download
    filename = 'adult.zip'
    filepath = os.path.join(save_dir, filename)
    if os.path.isfile(filepath) and not overwrite:
        print(f"'{filename}' already exists; skipping download")
    else:
        _ = utils.download_file_from_url(file_url=url, save_dir=save_dir)


# %% Load the original Adult dataset
def load_original() -> DataFrame:
    r"""
    Load the original Adult dataset [1]_.

    Assumes that the dataset is saved in '`DATASET_DIR`/_original',
    having file name 'adult.zip'. The loaded dataset is the
    concatenation 'adult.data' and 'adult.test'.

    Returns
    -------
    DataFrame
        Original Adult dataset. The index has name 'object', and
        columns 'attribute'.

    References
    ----------
    .. [1] Barry Becker, Ronny Kohavi. `Adult
       <https://doi.org/10.24432/C5XW20>`_. UCI Machine Learning
       Repository. 1996.

    """
    load_dir = os.path.join(DATASET_DIR, '_original')
    filename = 'adult.zip'
    filepath = os.path.join(load_dir, filename)
    print(f"Loading the original Adult dataset from '{filepath}'")
    with zipfile.ZipFile(filepath) as zf:
        train_frame = pd.read_csv(
            zf.open('adult.data'), sep=',', names=ATTRIBUTES,
            skipinitialspace=True, na_values=['?'],
            keep_default_na=False
            )
        test_frame = pd.read_csv(
            zf.open('adult.test'), sep=',', names=ATTRIBUTES,
            skipinitialspace=True, skiprows=1, na_values=['?'],
            keep_default_na=False
            )
    test_frame['annual-income'] = test_frame['annual-income'].str.rstrip('.')
    train_frame.index.name = test_frame.index.name = 'object'
    train_frame.columns.name = test_frame.columns.name = 'attribute'
    frame = pd.concat([train_frame, test_frame], ignore_index=True)
    frame.index.name = 'object'
    print(f"  shape: {frame.shape}")
    return frame


# %% Process the original Adult dataset
def process(
        export: bool = True, overwrite: bool = False
        ) -> tuple[str, DataFrame, Series, int]:
    r"""
    Process the original Adult dataset [1]_.

    Cleans the original dataset as per Le Quy (2022) [2]_. Uses the
    same non-sensitive attributes as Le Quy (2022) [2]_ except for
    'education' instead of which the equivalent attribute
    'education_num' is used. Uses the same sensitive attribute ('age')
    as Jiang (2022) [3]_. The non-sensitive continuous attributes are
    standardised, *i.e.*, zero mean and unit variance, following
    Ghadiri (2021) [4]_. The non-sensitive categorical attributes are
    one hot encoded. Optionally, the processed dataset is exported to
    '`DATASET_DIR`/adult.csv'.

    Parameters
    ----------
    export : bool, optional
        Whether to export the processed dataset. The default is True.
    overwrite : bool, optional
        Whether to overwrite existing files while exporting. The
        default is False.

    Raises
    ------
    FileExistsError
        If `export` is True, `overwrite` is False, and
        '`DATASET_DIR`/adult.csv' exists.

    Returns
    -------
    str
        Name of the processed dataset.
    DataFrame
        Dataframe corresponding to the non-sensitive attributes in the
        processed dataset.
    Series
        Series corresponding to the sensitive attribute in the
        processed dataset.
    int
        Number of non-sensitive attributes in the processed dataset.

    References
    ----------
    .. [1] Barry Becker, Ronny Kohavi. `Adult
       <https://doi.org/10.24432/C5XW20>`_. UCI Machine Learning
       Repository. 1996.
    .. [2] Tai Le Quy, Arjun Roy, Vasileios Iosifidis, Wenbin Zhang,
       Eirini Ntoutsi. `A survey on datasets for fairness-aware machine
       learning <https://doi.org/10.1002/widm.1452>`_. WIREs Data
       Mining and Knowledge Discovery. 2022.
    .. [3] Zhimeng Jiang, Xiaotian Han, Chao Fan, Fan Yang, Ali
       Mostafavi, Xia Hu. `Generalized Demographic Parity for Group
       Fairness <https://openreview.net/forum?id=YigKlMJwjye>`_. ICLR
       2022.
    .. [4] Mehrdad Ghadiri, Samira Samadi, Santosh Vempala. `Socially
       Fair k-Means Clustering
       <https://doi.org/10.1145/3442188.3445906>`_. FAccT 2021.

    """
    if export:
        filepath = os.path.join(DATASET_DIR, 'adult.csv')
        if os.path.isfile(filepath) and not overwrite:
            raise FileExistsError(f"'{filepath}' already exists")
    # load the original dataset
    original = load_original()
    assert set(NONSENSITIVE).issubset(original.columns)
    assert SENSITIVE in original.columns
    print("Processing the original Adult dataset")
    # drop rows with NaN as per Le Quy (2022)
    print("  Dropping rows with NaN as per Le Quy (2022)")
    frame = original.dropna()
    print(f"    shape: {frame.shape}")
    # rearrange to match the order of columns in `original`
    nonsensitive = [attr for attr in original.columns if attr in NONSENSITIVE]
    columns = nonsensitive + [SENSITIVE]
    # retain only required columns
    frame = frame.filter(columns, axis='columns')
    frame.columns.name = original.columns.name
    print("  Retaining only required attributes")
    print(f"    # non-sensitive attributes: {len(NONSENSITIVE)}")
    print(f"    sensitive attribute: '{SENSITIVE}'")
    print(f"    shape: {frame.shape}")
    # standardise continuous non-sensitive attributes
    continuous = [attr for attr in nonsensitive if attr not in CATEGORICAL]
    print("  Standardising continuous non-sensitive attributes")
    frame[continuous] = frame[continuous].apply(utils.standardise)
    # one hot encode categorical non-sensitive attributes
    categorical = [attr for attr in nonsensitive if attr in CATEGORICAL]
    if len(categorical) > 0:
        print("  One hot encoding categorical non-sensitive attributes")
        oha_frames = list()
        for attr in categorical:
            oha_frame = utils.one_hot_encode(ser=frame[attr])
            oha_frames.append(oha_frame)
            # get location of categorical attribute
            loc = columns.index(attr)
            # insert one hot encoded
            for oha in oha_frame.columns:
                loc += 1
                columns.insert(loc, oha)
        frame = pd.concat([frame] + oha_frames, axis='columns')
        frame = frame.filter(columns, axis='columns')
        print(f"    shape: {frame.shape}")
        # drop the original categorical attributes
        print("  Dropping original categorical non-sensitive attributes")
        frame.drop(columns=categorical, inplace=True)
        print(f"    shape: {frame.shape}")
    if export:
        # export
        print(f"Exporting the processed Adult dataset ({DATASET_NAME})"
              f" to '{filepath}'")
        frame.to_csv(filepath)
    X = frame.drop(SENSITIVE, axis='columns')
    s = frame.filter([SENSITIVE], axis='columns').squeeze()
    print(tabulate(
        {'dataset': [DATASET_NAME],
         '# objects': [len(X)],
         '# features': [len(X.columns)],
         '# nonsensitive attributes': [len(nonsensitive)],
         'sensitive attribute': [s.name]
         },
        headers='keys',
        tablefmt='rounded_outline',
        stralign='right'
        ))
    return DATASET_NAME, X, s, len(nonsensitive)


# %% Load the processed Adult dataset
def load() -> tuple[str, DataFrame, Series, int]:
    r"""
    Load the processed Adult dataset [1]_.

    Assumes that the dataset is saved in '`DATASET_DIR`' having file
    name 'adult.csv'. Has the same non-sensitive attributes as Le Quy
    (2022) [2]_ except for 'education' instead of which the equivalent
    attribute 'education_num' is used. Has the same sensitive attribute
    ('age') as Jiang (2022) [3]_. The non-sensitive continuous
    attributes are standardised, *i.e.*, zero mean and unit variance,
    following Ghadiri (2021) [4]_. The non-sensitive categorical
    attributes are one hot encoded.

    Returns
    -------
    str
        Name of the processed dataset.
    DataFrame
        Dataframe corresponding to the non-sensitive attributes in the
        processed dataset.
    Series
        Series corresponding to the sensitive attribute in the
        processed dataset.
    int
        Number of non-sensitive attributes in the processed dataset.

    References
    ----------
    .. [1] Barry Becker, Ronny Kohavi. `Adult
       <https://doi.org/10.24432/C5XW20>`_. UCI Machine Learning
       Repository. 1996.
    .. [2] Tai Le Quy, Arjun Roy, Vasileios Iosifidis, Wenbin Zhang,
       Eirini Ntoutsi. `A survey on datasets for fairness-aware machine
       learning <https://doi.org/10.1002/widm.1452>`_. WIREs Data
       Mining and Knowledge Discovery. 2022.
    .. [3] Zhimeng Jiang, Xiaotian Han, Chao Fan, Fan Yang, Ali
       Mostafavi, Xia Hu. `Generalized Demographic Parity for Group
       Fairness <https://openreview.net/forum?id=YigKlMJwjye>`_. ICLR
       2022.
    .. [4] Mehrdad Ghadiri, Samira Samadi, Santosh Vempala. `Socially
       Fair k-Means Clustering
       <https://doi.org/10.1145/3442188.3445906>`_. FAccT 2021.

    """
    filepath = os.path.join(DATASET_DIR, 'adult.csv')
    print(f"Loading the processed Adult dataset ({DATASET_NAME}) from"
          f" '{filepath}'")
    frame = pd.read_csv(filepath, index_col=0)
    frame.columns.name = 'attribute'
    X = frame.drop(SENSITIVE, axis='columns')
    s = frame.filter([SENSITIVE], axis='columns').squeeze()
    n_nonsensitive = utils.count_nonsensitive(X=X)
    print(tabulate(
        {'dataset': [DATASET_NAME],
         '# objects': [len(X)],
         '# features': [len(X.columns)],
         '# nonsensitive attributes': [n_nonsensitive],
         'sensitive attribute': [s.name]
         },
        headers='keys',
        tablefmt='rounded_outline',
        stralign='right'
        ))
    return DATASET_NAME, X, s, n_nonsensitive


# %% END OF FILE
