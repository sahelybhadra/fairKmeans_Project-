#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Utilities for processing the COMPAS dataset [1]_.

The attributes for the original dataset are taken from Le Quy (2022)
[2]_. For the processed version, the original dataset is cleaned
following the `analysis
<https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb>`_
by Angwin (2016) [1]_. The non-sensitive attributes are selected
following Le Quy (2022) [2]_. We use 'age' as the sensitive attribute
following the `analysis
<https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb>`_
by Angwin (2016) [1]_ (and seemingly in Grari (2021) [3]_).

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
    Sensitive attribute (seemingly used in Grari (2021) [3]_).

Routine Listings
----------------
download()
    Download the original COMPAS dataset.
load_original()
    Load the original COMPAS dataset.
process()
    Process the original COMPAS dataset.
load()
    Load the processed COMPAS dataset.

References
----------
.. [1] Julia Angwin, Jeff Larson, Surya Mattu, Lauren Kirchner.
   `Machine Bias <https://github.com/propublica/compas-analysis>`_.
   ProPublica. 2016.
.. [2] Tai Le Quy, Arjun Roy, Vasileios Iosifidis, Wenbin Zhang, Eirini
   Ntoutsi. `A survey on datasets for fairness-aware machine learning
   <https://doi.org/10.1002/widm.1452>`_. WIREs Data Mining and
   Knowledge Discovery. 2022.
.. [3] Vincent Grari, Oualid El Hajouji, Sylvain Lamprier, Marcin
   Detyniecki. `Learning Unbiased Representations via Rényi
   Minimization <https://doi.org/10.1007/978-3-030-86520-7\_46>`_. ECML
   PKDD 2021.

"""


# %% Libraries
import os
import pandas as pd
from pandas import DataFrame, Series
from tabulate import tabulate

from . import utils


# %% Paths
# Subdirectory that would contain the COMPAS dataset
DATASET_DIR: str = 'data/datasets/compas'


# %% Dataset name
DATASET_NAME: str = 'compas'


# %% Attributes taken from Le Quy (2022)
ATTRIBUTES: list[str] = [
    'name', 'first', 'last', 'compas_screening_date', 'sex', 'dob',
    'age', 'age_cat', 'race', 'juv_fel_count', 'decile_score',
    'juv_misd_count', 'juv_other_count', 'priors_count',
    'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
    'c_case_number', 'c_offense_date', 'c_arrest_date',
    'c_days_from_compas', 'c_charge_degree', 'c_charge_desc',
    'is_recid', 'r_case_number', 'r_charge_degree',
    'r_days_from_arrest', 'r_offense_date', 'r_charge_desc',
    'r_jail_in', 'r_jail_out', 'violent_recid', 'is_violent_recid',
    'vr_case_number', 'vr_charge_degree', 'vr_offense_date',
    'vr_charge_desc', 'type_of_assessment', 'decile_score.1',
    'score_text', 'screening_date', 'v_type_of_assessment',
    'v_decile_score', 'v_score_text', 'v_screening_date', 'in_custody',
    'out_custody', 'priors_count.1', 'start', 'end', 'event',
    'two_year_recid'
    ]


# %% Categorical attributes taken from Le Quy (2022)
CATEGORICAL: list[str] = [
    'name', 'first', 'last', 'compas_screening_date', 'sex', 'dob',
    'age_cat', 'race', 'c_jail_in', 'c_jail_out', 'c_case_number',
    'c_offense_date', 'c_arrest_date', 'c_charge_degree',
    'c_charge_desc', 'is_recid', 'r_case_number', 'r_charge_degree',
    'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out',
    'is_violent_recid', 'vr_case_number', 'vr_charge_degree',
    'vr_offense_date', 'vr_charge_desc', 'type_of_assessment',
    'score_text', 'screening_date', 'v_type_of_assessment',
    'v_score_text', 'v_screening_date', 'in_custody', 'out_custody',
    'event', 'two_year_recid'
    ]


# %% Non-sensitive attributes used in Le Quy (2022)
NONSENSITIVE: list[str] = [
    'juv_fel_count', 'decile_score', 'juv_misd_count',
    'juv_other_count', 'priors_count', 'c_charge_degree',
    'v_decile_score'
    ]


# %% Sensitive attribute (seemingly used in Grari (2021))
SENSITIVE: str = 'age'


# %% Download the original COMPAS dataset
def download(
        url: str = 'https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv',
        overwrite: bool = False
        ):
    r"""
    Download the original COMPAS dataset [1]_.

    Assumes that the data file name is 'compas-scores-two-years.csv'.
    Saves the downloaded dataset in '`DATASET_DIR`/_original'.

    Parameters
    ----------
    url : str, optional
        URL of the dataset. The default is
        'https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv'.
    overwrite : bool, optional
        Whether to overwrite existing files. The default is False.

    Returns
    -------
    None.

    References
    ----------
    .. [1] Julia Angwin, Jeff Larson, Surya Mattu, Lauren Kirchner.
       `Machine Bias <https://github.com/propublica/compas-analysis>`_.
       ProPublica. 2016.

    """
    print("Preparing to download the original COMPAS dataset")
    save_dir = os.path.join(DATASET_DIR, '_original')
    # download
    filename = 'compas-scores-two-years.csv'
    filepath = os.path.join(save_dir, filename)
    if os.path.isfile(filepath) and not overwrite:
        print(f"'{filename}' already exists; skipping download")
    else:
        _ = utils.download_file_from_url(file_url=url, save_dir=save_dir)


# %% Load the original COMPAS dataset
def load_original() -> DataFrame:
    r"""
    Load the original COMPAS dataset [1]_.

    Assumes that the dataset is saved in '`DATASET_DIR`/_original',
    having file name 'compas-scores-two-years.csv'.

    Returns
    -------
    DataFrame
        Original COMPAS dataset. The index has name 'object', and
        columns 'attribute'.

    References
    ----------
    .. [1] Julia Angwin, Jeff Larson, Surya Mattu, Lauren Kirchner.
       `Machine Bias <https://github.com/propublica/compas-analysis>`_.
       ProPublica. 2016.

    """
    load_dir = os.path.join(DATASET_DIR, '_original')
    filename = 'compas-scores-two-years.csv'
    filepath = os.path.join(load_dir, filename)
    print(f"Loading the original COMPAS dataset from '{filepath}'")
    frame = pd.read_csv(filepath, sep=',', index_col=0)
    assert frame.columns.to_list() == ATTRIBUTES
    frame.index.name = 'object'
    frame.columns.name = 'attribute'
    print(f"  shape: {frame.shape}")
    return frame


# %% Process the original COMPAS dataset
def process(
        export: bool = True, overwrite: bool = False
        ) -> tuple[str, DataFrame, Series, int]:
    r"""
    Process the original COMPAS dataset [1]_.

    Cleans the original dataset following the `analysis
    <https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb>`_
    by Angwin (2016) [1]_. Uses the same non-sensitive attributes as Le
    Quy (2022) [2]_ and the sensitive attribute is 'age' (seemingly
    used in Grari (2021) [3]_). The non-sensitive continuous attributes
    are standardised, *i.e.*, zero mean and unit variance, following
    Ghadiri (2021) [4]_. The non-sensitive categorical attributes are
    one hot encoded. Optionally, the processed dataset is exported to
    '`DATASET_DIR`/compas.csv'.

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
        '`DATASET_DIR`/compas.csv' exists.

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
    .. [1] Julia Angwin, Jeff Larson, Surya Mattu, Lauren Kirchner.
       `Machine Bias <https://github.com/propublica/compas-analysis>`_.
       ProPublica. 2016.
    .. [2] Tai Le Quy, Arjun Roy, Vasileios Iosifidis, Wenbin Zhang,
       Eirini Ntoutsi. `A survey on datasets for fairness-aware machine
       learning <https://doi.org/10.1002/widm.1452>`_. WIREs Data
       Mining and Knowledge Discovery. 2022.
    .. [3] Vincent Grari, Oualid El Hajouji, Sylvain Lamprier, Marcin
       Detyniecki. `Learning Unbiased Representations via Rényi
       Minimization <https://doi.org/10.1007/978-3-030-86520-7\_46>`_.
       ECML PKDD 2021.
    .. [4] Mehrdad Ghadiri, Samira Samadi, Santosh Vempala. `Socially
       Fair k-Means Clustering
       <https://doi.org/10.1145/3442188.3445906>`_. FAccT 2021.

    """
    if export:
        filepath = os.path.join(DATASET_DIR, 'compas.csv')
        if os.path.isfile(filepath) and not overwrite:
            raise FileExistsError(f"'{filepath}' already exists")
    # load the original dataset
    original = load_original()
    assert set(NONSENSITIVE).issubset(original.columns)
    assert SENSITIVE in original.columns
    print("Processing the original COMPAS dataset")
    # filter rows as per Angwin (2016)
    print("  Filtering rows as per Angwin (2016)")
    # retain rows where -30 ≤ 'days_b_screening_arrest' ≤ 30
    index = (-30 <= original['days_b_screening_arrest'])
    index = index & (original['days_b_screening_arrest'] <= 30)
    # drop rows where 'is_recid' is -1
    index = index & (original['is_recid'] != -1)
    # drop rows where 'c_charge_degree' is 'O'
    index = index & (original['c_charge_degree'] != 'O')
    # drop rows where 'score_text' is NaN
    index = index & (original['score_text'].notna())
    frame = original[index]
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
        print(f"Exporting the processed COMPAS dataset"
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


# %% Load the processed COMPAS dataset
def load() -> tuple[str, DataFrame, Series, int]:
    r"""
    Load the processed COMPAS dataset [1]_.

    Assumes that the dataset is saved in '`DATASET_DIR`' having file
    name 'compas.csv'. Has the same non-sensitive attributes as Le Quy
    (2022) [2]_ and the sensitive attribute is 'age'. The non-sensitive
    continuous attributes are standardised, *i.e.*, zero mean and unit
    variance, following Ghadiri (2021) [3]_. The non-sensitive
    categorical attributes are one hot encoded.

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
    .. [1] Julia Angwin, Jeff Larson, Surya Mattu, Lauren Kirchner.
       `Machine Bias <https://github.com/propublica/compas-analysis>`_.
       ProPublica. 2016.
    .. [2] Tai Le Quy, Arjun Roy, Vasileios Iosifidis, Wenbin Zhang,
       Eirini Ntoutsi. `A survey on datasets for fairness-aware machine
       learning <https://doi.org/10.1002/widm.1452>`_. WIREs Data
       Mining and Knowledge Discovery. 2022.
    .. [3] Mehrdad Ghadiri, Samira Samadi, Santosh Vempala. `Socially
       Fair k-Means Clustering
       <https://doi.org/10.1145/3442188.3445906>`_. FAccT 2021.

    """
    filepath = os.path.join(DATASET_DIR, 'compas.csv')
    print(f"Loading the processed COMPAS dataset ({DATASET_NAME}) from"
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
