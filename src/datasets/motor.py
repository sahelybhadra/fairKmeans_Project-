#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Utilities for processing the Motor Insurance dataset [1]_.

The attributes for the original dataset are taken from the
'Pricing_ENG.pdf' file. For the processed version, the sensitive
attribute ('Age') following Grari (2020) [2]_.

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
    Non-sensitive attributes.
SENSITIVE : str
    Sensitive attribute used in Grari (2020) [2]_.

Routine Listings
----------------
download()
    Download the original Motor Insurance dataset.
load_original()
    Load the original Motor Insurance dataset.
process()
    Process the original Motor Insurance dataset.
load()
    Load the processed Motor Insurance dataset.

References
----------
.. [1] Arthur Charpentier, Romuald Elie, Jérémie Jakubowicz. `Pricing
   Game <https://freakonometrics.hypotheses.org/20191>`_. French
   Institute of Actuaries. 2015.
.. [2] Vincent Grari, Sylvain Lamprier, Marcin Detyniecki.
   `Fairness-Aware Neural Rényi Minimization for Continuous Features
   <https://doi.org/10.24963/ijcai.2020/313>`_. IJCAI 2020.

"""


# %% Libraries
import os
import pandas as pd
from pandas import DataFrame, Series
from tabulate import tabulate

from . import utils


# %% Paths
# Subdirectory that would contain the Motor Insurance dataset
DATASET_DIR: str = 'data/datasets/motor'


# %% Dataset name
DATASET_NAME: str = 'motor'


# %% Attributes taken from 'Pricing_ENG.pdf'
ATTRIBUTES: list[str] = [
    'PolNum', 'CalYear', 'Gender', 'Type', 'Category', 'Occupation',
    'Age', 'Group1', 'Bonus', 'Poldur', 'Value', 'Adind', 'SubGroup2',
    'Group2', 'Density'
    ]


# %% Categorical attributes taken from 'Pricing_ENG.pdf'
CATEGORICAL: list[str] = [
    'PolNum', 'CalYear', 'Gender', 'Type', 'Category', 'Occupation',
    'Group1', 'Adind', 'SubGroup2', 'Group2'
    ]


# %% Non-sensitive attributes
NONSENSITIVE: list[str] = [
    'Type', 'Category', 'Occupation', 'Group1', 'Bonus', 'Poldur',
    'Value', 'Adind', 'SubGroup2', 'Group2', 'Density'
    ]


# %% Sensitive attribute used in Grari (2020)
SENSITIVE: str = 'Age'


# %% Download the original Motor Insurance dataset
def download(
        data_url: str = 'http://freakonometrics.free.fr/pricing.csv',
        metadata_url: str = 'https://freakonometrics.hypotheses.org/files/2015/08/Pricing_ENG.pdf',
        overwrite: bool = False
        ):
    r"""
    Download the original Motor Insurance dataset [1]_.

    Assumes that the data file name is 'pricing.csv' and metadata file
    name is 'Pricing_ENG.pdf'. Saves the downloaded dataset in
    '`DATASET_DIR`/_original'.

    Parameters
    ----------
    data_url : str, optional
        URL of the dataset. The default is
        'http://freakonometrics.free.fr/pricing.csv'.
    metadata_url : str, optional
        URL of the metadata. The default is
        'https://freakonometrics.hypotheses.org/files/2015/08/Pricing_ENG.pdf'.
    overwrite : bool, optional
        Whether to overwrite existing files. The default is False.

    Returns
    -------
    None.

    References
    ----------
    .. [1] Arthur Charpentier, Romuald Elie, Jérémie Jakubowicz.
       `Pricing Game <https://freakonometrics.hypotheses.org/20191>`_.
       French Institute of Actuaries. 2015.

    """
    print("Preparing to download the original Motor Insurance dataset")
    save_dir = os.path.join(DATASET_DIR, '_original')
    # download
    urls = {'pricing.csv': data_url, 'Pricing_ENG.pdf': metadata_url}
    for filename, url in urls.items():
        filepath = os.path.join(save_dir, filename)
        if os.path.isfile(filepath) and not overwrite:
            print(f"'{filename}' already exists; skipping download")
        else:
            _ = utils.download_file_from_url(file_url=url, save_dir=save_dir)


# %% Load the original Motor Insurance dataset
def load_original() -> DataFrame:
    r"""
    Load the original Motor Insurance dataset [1]_.

    Assumes that the dataset is saved in '`DATASET_DIR`/_original',
    having file name 'pricing.csv'.

    Returns
    -------
    DataFrame
        Original Motor Insurance dataset. The index has name 'object',
        and columns 'attribute'.

    References
    ----------
    .. [1] Arthur Charpentier, Romuald Elie, Jérémie Jakubowicz.
       `Pricing Game <https://freakonometrics.hypotheses.org/20191>`_.
       French Institute of Actuaries. 2015.

    """
    load_dir = os.path.join(DATASET_DIR, '_original')
    filename = 'pricing.csv'
    filepath = os.path.join(load_dir, filename)
    print(f"Loading the original Motor Insurance dataset from '{filepath}'")
    frame = pd.read_csv(filepath, sep=';', index_col=0)
    assert frame.columns.to_list() == ATTRIBUTES
    frame.index.name = 'object'
    frame.columns.name = 'attribute'
    print(f"  shape: {frame.shape}")
    return frame


# %% Process the original Motor Insurance dataset
def process(
        export: bool = True, overwrite: bool = False
        ) -> tuple[str, DataFrame, Series, int]:
    r"""
    Process the original Motor Insurance dataset [1]_.

    Uses the same sensitive attribute ('Age') as Grari (2020) [2]_. The
    non-sensitive continuous attributes are standardised, *i.e.*, zero
    mean and unit variance, following Ghadiri (2021) [3]_. The
    non-sensitive categorical attributes are one hot encoded.
    Optionally, the processed dataset is exported to
    '`DATASET_DIR`/motor.csv'.

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
        '`DATASET_DIR`/motor.csv' exists.

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
    .. [1] Arthur Charpentier, Romuald Elie, Jérémie Jakubowicz.
       `Pricing Game <https://freakonometrics.hypotheses.org/20191>`_.
       French Institute of Actuaries. 2015.
    .. [2] Vincent Grari, Sylvain Lamprier, Marcin Detyniecki.
       `Fairness-Aware Neural Rényi Minimization for Continuous
       Features <https://doi.org/10.24963/ijcai.2020/313>`_. IJCAI
       2020.
    .. [3] Mehrdad Ghadiri, Samira Samadi, Santosh Vempala. `Socially
       Fair k-Means Clustering
       <https://doi.org/10.1145/3442188.3445906>`_. FAccT 2021.

    """
    if export:
        filepath = os.path.join(DATASET_DIR, 'motor.csv')
        if os.path.isfile(filepath) and not overwrite:
            raise FileExistsError(f"'{filepath}' already exists")
    # load the original dataset
    original = load_original()
    assert set(NONSENSITIVE).issubset(original.columns)
    assert SENSITIVE in original.columns
    print("Processing the original Motor Insurance dataset")
    # rearrange to match the order of columns in `original`
    nonsensitive = [attr for attr in original.columns if attr in NONSENSITIVE]
    columns = nonsensitive + [SENSITIVE]
    # retain only required columns
    frame = original.filter(columns, axis='columns')
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
        print(f"Exporting the processed Motor Insurance dataset"
              f" ({DATASET_NAME}) to '{filepath}'")
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


# %% Load the processed Motor Insurance dataset
def load() -> tuple[str, DataFrame, Series, int]:
    r"""
    Load the processed Motor Insurance dataset [1]_.

    Assumes that the dataset is saved in '`DATASET_DIR`' having file
    name 'motor.csv'. Has the same sensitive attribute ('Age') as Grari
    (2020) [2]_. The non-sensitive continuous attributes are
    standardised, *i.e.*, zero mean and unit variance, following
    Ghadiri (2021) [3]_. The non-sensitive categorical attributes are
    one hot encoded.

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
    .. [1] Arthur Charpentier, Romuald Elie, Jérémie Jakubowicz.
       `Pricing Game <https://freakonometrics.hypotheses.org/20191>`_.
       French Institute of Actuaries. 2015.
    .. [2] Vincent Grari, Sylvain Lamprier, Marcin Detyniecki.
       `Fairness-Aware Neural Rényi Minimization for Continuous
       Features <https://doi.org/10.24963/ijcai.2020/313>`_. IJCAI
       2020.
    .. [3] Mehrdad Ghadiri, Samira Samadi, Santosh Vempala. `Socially
       Fair k-Means Clustering
       <https://doi.org/10.1145/3442188.3445906>`_. FAccT 2021.

    """
    filepath = os.path.join(DATASET_DIR, 'motor.csv')
    print(f"Loading the processed Motor Insurance dataset"
          f" ({DATASET_NAME}) from '{filepath}'")
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
