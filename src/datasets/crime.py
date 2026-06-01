#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for processing the Communities and Crime dataset [1]_.

The attributes for the original dataset are taken from its
'communities.names' file. For the processed version, the non-sensitive
attributes are selected following Le Quy (2022) [2]_, and sensitive
attribute ('racepctblack') following Jiang (2022) [3]_.

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
    Download the original Communities and Crime dataset.
load_original()
    Load the original Communities and Crime dataset.
process()
    Process the original Communities and Crime dataset.
load()
    Load the processed Communities and Crime dataset.

References
----------
.. [1] Michael Redmond. `Communities and Crime
   <https://doi.org/10.24432/C53W3X>`_. UCI Machine Learning
   Repository. 2009.
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
# Subdirectory that would contain the Communities and Crime dataset
DATASET_DIR: str = 'data/datasets/crime'


# %% Dataset name
DATASET_NAME: str = 'crime'


# %% Attributes taken from 'communities.names'
ATTRIBUTES: list[str] = [
    'state', 'county', 'community', 'communityname', 'fold',
    'population', 'householdsize', 'racepctblack', 'racePctWhite',
    'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct12t29',
    'agePct16t24', 'agePct65up', 'numbUrban', 'pctUrban', 'medIncome',
    'pctWWage', 'pctWFarmSelf', 'pctWInvInc', 'pctWSocSec',
    'pctWPubAsst', 'pctWRetire', 'medFamInc', 'perCapInc',
    'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap',
    'OtherPerCap', 'HispPerCap', 'NumUnderPov', 'PctPopUnderPov',
    'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore', 'PctUnemployed',
    'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu',
    'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr',
    'FemalePctDiv', 'TotalPctDiv', 'PersPerFam', 'PctFam2Par',
    'PctKids2Par', 'PctYoungKids2Par', 'PctTeen2Par',
    'PctWorkMomYoungKids', 'PctWorkMom', 'NumIlleg', 'PctIlleg',
    'NumImmig', 'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8',
    'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8',
    'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell',
    'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous',
    'PersPerOwnOccHous', 'PersPerRentOccHous', 'PctPersOwnOccup',
    'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant',
    'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded',
    'PctVacMore6Mos', 'MedYrHousBuilt', 'PctHousNoPhone',
    'PctWOFullPlumb', 'OwnOccLowQuart', 'OwnOccMedVal',
    'OwnOccHiQuart', 'RentLowQ', 'RentMedian', 'RentHighQ', 'MedRent',
    'MedRentPctHousInc', 'MedOwnCostPctInc', 'MedOwnCostPctIncNoMtg',
    'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState',
    'PctSameHouse85', 'PctSameCity85', 'PctSameState85',
    'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps',
    'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop',
    'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol',
    'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian',
    'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz',
    'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans',
    'PolicCars', 'PolicOperBudg', 'LemasPctPolicOnPatr',
    'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop',
    'ViolentCrimesPerPop'
    ]


# %% Categorical attributes taken from 'communities.names'
CATEGORICAL: list[str] = [
    'state', 'county', 'community', 'communityname', 'fold'
    ]


# %% Non-sensitive attributes used in Le Quy (2022)
NONSENSITIVE: list[str] = [
    'pctWInvInc', 'pctWPubAsst', 'NumUnderPov', 'PctPopUnderPov',
    'PctUnemployed', 'MalePctDivorce', 'FemalePctDiv', 'TotalPctDiv',
    'PersPerFam', 'PctKids2Par', 'PctYoungKids2Par', 'PctTeen2Par',
    'NumIlleg', 'PctIlleg', 'PctPersOwnOccup', 'HousVacant',
    'PctHousOwnOcc', 'PctVacantBoarded', 'NumInShelters', 'NumStreet'
    ]


# %% Sensitive attribute used in Jiang (2022)
SENSITIVE: str = 'racepctblack'


# %% Download the original Communities and Crime dataset
def download(
        url: str = 'https://archive.ics.uci.edu/static/public/183/communities+and+crime.zip',
        overwrite: bool = False
        ):
    """
    Download the original Communities and Crime dataset [1]_.

    Assumes that the data file name is 'communities+and+crime.zip'.
    Saves the downloaded dataset in '`DATASET_DIR`/_original'.

    Parameters
    ----------
    url : str, optional
        URL of the dataset. The default is
        'https://archive.ics.uci.edu/static/public/183/communities+and+crime.zip'.
    overwrite : bool, optional
        Whether to overwrite existing files. The default is False.

    Returns
    -------
    None.

    References
    ----------
    .. [1] Michael Redmond. `Communities and Crime
       <https://doi.org/10.24432/C53W3X>`_. UCI Machine Learning
       Repository. 2009.

    """
    print("Preparing to download the original Communities and Crime dataset")
    save_dir = os.path.join(DATASET_DIR, '_original')
    # download
    filename = 'communities+and+crime.zip'
    filepath = os.path.join(save_dir, filename)
    if os.path.isfile(filepath) and not overwrite:
        print(f"'{filename}' already exists; skipping download")
    else:
        _ = utils.download_file_from_url(file_url=url, save_dir=save_dir)


# %% Load the original Communities and Crime dataset
def load_original() -> DataFrame:
    r"""
    Load the original Communities and Crime dataset [1]_.

    Assumes that the dataset is saved in '`DATASET_DIR`/_original',
    having file name 'communities+and+crime.zip'.

    Returns
    -------
    DataFrame
        Original Communities and Crime dataset. The index has name
        'object', and columns 'attribute'.

    References
    ----------
    .. [1] Michael Redmond. `Communities and Crime
       <https://doi.org/10.24432/C53W3X>`_. UCI Machine Learning
       Repository. 2009.

    """
    load_dir = os.path.join(DATASET_DIR, '_original')
    filename = 'communities+and+crime.zip'
    filepath = os.path.join(load_dir, filename)
    print(f"Loading the original Communities and"
          f" Crime dataset from '{filepath}'")
    with zipfile.ZipFile(filepath) as zf:
        frame = pd.read_csv(
            zf.open('communities.data'), sep=',',
            header=None, names=ATTRIBUTES, na_values=['?']
            )
    frame.index.name = 'object'
    frame.columns.name = 'attribute'
    print(f"  shape: {frame.shape}")
    return frame


# %% Process the original Communities and Crime dataset
def process(
        export: bool = True, overwrite: bool = False
        ) -> tuple[str, DataFrame, Series, int]:
    """
    Process the original Communities and Crime dataset [1]_.

    Uses the same non-sensitive attributes as Le Quy (2022) [2]_ and
    sensitive attribute ('racepctblack') as Jiang (2022) [3]_.
    Optionally, the processed dataset is exported to
    '`DATASET_DIR`/crime.csv'.

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
        '`DATASET_DIR`/crime.csv' exists.

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
    .. [1] Michael Redmond. `Communities and Crime
       <https://doi.org/10.24432/C53W3X>`_. UCI Machine Learning
       Repository. 2009.
    .. [2] Tai Le Quy, Arjun Roy, Vasileios Iosifidis, Wenbin Zhang,
       Eirini Ntoutsi. `A survey on datasets for fairness-aware machine
       learning <https://doi.org/10.1002/widm.1452>`_. WIREs Data
       Mining and Knowledge Discovery. 2022.
    .. [3] Zhimeng Jiang, Xiaotian Han, Chao Fan, Fan Yang, Ali
       Mostafavi, Xia Hu. `Generalized Demographic Parity for Group
       Fairness <https://openreview.net/forum?id=YigKlMJwjye>`_. ICLR
       2022.

    """
    if export:
        filepath = os.path.join(DATASET_DIR, 'crime.csv')
        if os.path.isfile(filepath) and not overwrite:
            raise FileExistsError(f"'{filepath}' already exists")
    # load the original dataset
    original = load_original()
    assert set(NONSENSITIVE).issubset(original.columns)
    assert SENSITIVE in original.columns
    print("Processing the original Communities and Crime dataset")
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
    if export:
        # export
        print(f"Exporting the processed Communities and Crime"
              f" dataset ({DATASET_NAME}) to '{filepath}'")
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


# %% Load the processed Communities and Crime dataset
def load() -> tuple[str, DataFrame, Series, int]:
    """
    Load the processed Communities and Crime dataset [1]_.

    Assumes that the dataset is saved in '`DATASET_DIR`' having file
    name 'crime.csv'. Has the same non-sensitive attributes as Le Quy
    (2022) [2]_ and sensitive attribute ('racepctblack') as Jiang
    (2022) [3]_.

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
    .. [1] Michael Redmond. `Communities and Crime
       <https://doi.org/10.24432/C53W3X>`_. UCI Machine Learning
       Repository. 2009.
    .. [2] Tai Le Quy, Arjun Roy, Vasileios Iosifidis, Wenbin Zhang,
       Eirini Ntoutsi. `A survey on datasets for fairness-aware machine
       learning <https://doi.org/10.1002/widm.1452>`_. WIREs Data
       Mining and Knowledge Discovery. 2022.
    .. [3] Zhimeng Jiang, Xiaotian Han, Chao Fan, Fan Yang, Ali
       Mostafavi, Xia Hu. `Generalized Demographic Parity for Group
       Fairness <https://openreview.net/forum?id=YigKlMJwjye>`_. ICLR
       2022.

    """
    filepath = os.path.join(DATASET_DIR, 'crime.csv')
    print(f"Loading the processed Communities and Crime"
          f" dataset ({DATASET_NAME}) from '{filepath}'")
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
