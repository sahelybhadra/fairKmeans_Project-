#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Evaluation metrics.

Routine Listings
----------------
bera_generalised_balance()
    Bera (2019) [1]_'s generalised balance.
evaluate()
    Evaluate against all metrics.
kmeans_objective()
    :math:`k`-means objective.
max_ks_stat()
    Maximum Kolmogorov-Smirnov statistic.
max_emd()
    Maximum earth mover's distance.
modified_abraham_deviation()
    Modified Abraham (2020) [2]_'s deviation.
pooling_window_loss()
    Pooling window loss.
ziko_fairness_error()
    Ziko (2021) [3]_'s fairness error.

References
----------
.. [1] Suman Bera, Deeparnab Chakrabarty, Nicolas Flores, Maryam
   Negahbani. `Fair Algorithms for Clustering
   <https://proceedings.neurips.cc/paper/2019/hash/fc192b0c0d270dbf41870a63a8c76c2f-Abstract.html>`_.
   NeurIPS 2019.
.. [2] Savitha Sam Abraham, Deepak P, Sowmya S. Sundaram. `Fairness in
   Clustering with Multiple Sensitive Attributes
   <https://doi.org/10.5441/002/edbt.2020.26>`_. EDBT 2020.
.. [3] Imtiaz Masud Ziko, Jing Yuan, Eric Granger, Ismail Ben Ayed.
   `Variational Fair Clustering
   <https://doi.org/10.1609/aaai.v35i12.17336>`_. AAAI 2021.

"""


# %% Libraries
import numpy as np
from pandas import DataFrame, Series

from .algorithms import abraham2020, fairkmeans_prc, lloyd


# %% Bera (2019)'s generalised balance -- higher is better
def bera_generalised_balance(s: Series, c: Series) -> float:
    r"""
    Compute Bera (2019) [1]_'s generalised balance.

    Higher is better.

    Parameters
    ----------
    s : Series
        Sensitive attribute values.
    c : Series
        Cluster assignment.

    Returns
    -------
    float
        Bera (2019)'s generalised balance.

    Notes
    -----
    Bera (2019, §4) [1]_'s generalised balance for the cluster assignment
    :math:`\mathcal{C}` is computed as

    .. math::
        \min_{C \in \mathcal{C}}\
        \min_{S \in \mathcal{S}}\
        \min\bigg(\frac{f_X(S)}{f_C(S)}, \frac{f_C(S)}{f_X(S)}\bigg)

    where :math:`\mathcal{S}` is the set of sensitive attribute values,
    and :math:`f_X` and :math:`f_C` are respectively the ratios of a
    sensitive attribute value in the dataset :math:`X` and cluster
    :math:`C` given by

    .. math::
        f_X(S) =
            \frac{|X \cap S|}{|X|}
            \qquad \qquad
            f_C(S) = \frac{|C \cap S|}{|C|}

    References
    ----------
    .. [1] Suman Bera, Deeparnab Chakrabarty, Nicolas Flores, Maryam
       Negahbani. `Fair Algorithms for Clustering
       <https://proceedings.neurips.cc/paper/2019/hash/fc192b0c0d270dbf41870a63a8c76c2f-Abstract.html>`_.
       NeurIPS 2019.

    """
    f_X = s.value_counts(normalize=True, sort=False)
    f_C = s.groupby(c, sort=False).value_counts(normalize=True, sort=False)
    X_by_C = f_X.divide(f_C, level=s.name)
    C_by_X = np.reciprocal(X_by_C)
    min_ = np.minimum(X_by_C, C_by_X)
    balance_C = min_.groupby(level=c.name, sort=False).min()
    return balance_C.min()


# %% Evaluate against all metrics
def evaluate(
        X: DataFrame, n_nonsensitive: int, s: Series, c: Series,
        centroids: DataFrame, n_clusters: int, window_size: int
        ) -> Series:
    r"""
    Evaluate against all metrics.

    Parameters
    ----------
    X : DataFrame
        Dataset.
    n_nonsensitive : int
        Number of non-sensitive attributes.
    s : Series
        Sensitive attribute values.
    c : Series
        Cluster assignment.
    centroids : DataFrame
        Centroids.
    n_clusters : int
        Number of clusters.
    window_size : int
        Pooling window size.

    Returns
    -------
    Series
        Evaluation scores. Has the following index:

        +------------------------------------------+
        | k-means objective                        |
        +------------------------------------------+
        | max ks statistic                         |
        +------------------------------------------+
        | max emd                                  |
        +------------------------------------------+
        | pooling window loss (size=`window_size`) |
        +------------------------------------------+
        | Modified Abraham (2020)'s deviation      |
        +------------------------------------------+
        | Ziko (2021)'s fairness error             |
        +------------------------------------------+
        | Bera (2019)'s generalised balance        |
        +------------------------------------------+

    """
    scores = Series({
        "k-means objective": kmeans_objective(
            X=X, n_nonsensitive=n_nonsensitive, c=c,
            centroids=centroids
            ),
        "max ks statistic": max_ks_stat(s=s, c=c),
        "max emd": max_emd(s=s, c=c),
        f"pooling window loss (size={window_size})": pooling_window_loss(
            s=s, c=c, window_size=window_size
            ),
        "Modified Abraham (2020)'s deviation": modified_abraham_deviation(
            s=s, c=c, n_clusters=n_clusters
            ),
        "Ziko (2021)'s fairness error": ziko_fairness_error(s=s, c=c),
        "Bera (2019)'s generalised balance": bera_generalised_balance(s=s, c=c)
        }, name='score')
    scores.index.name = 'metric'
    return scores


# %% K-means objective -- lower is better
def kmeans_objective(
        X: DataFrame, n_nonsensitive: int, c: Series,
        centroids: DataFrame
        ) -> float:
    r"""
    Compute :math:`k`-means objective.

    Lower is better.

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
        :math:`k`-means objective.

    Notes
    -----
    The :math:`k`-means objective is computed as

    .. math::
        \frac{1}{\textit{n_nonsensitive} \times |X|}
        \times \sum_{C \in \mathcal{C}}\ \sum_{x \in C}\ {||x - \mu_C||}^2

    where :math:`\mathcal{C}` is the cluster assignment and
    :math:`\mu_C` is the centroid of cluster :math:`C`.

    """
    return lloyd.compute_objective(
        X=X, n_nonsensitive=n_nonsensitive, c=c, centroids=centroids
        )


# %% Maximum Kolmogorov-Smirnov statistic -- lower is better
def max_ks_stat(s: Series, c: Series) -> float:
    r"""
    Compute maximum Kolmogorov-Smirnov statistic [1]_.

    Lower is better.

    Parameters
    ----------
    s : Series
        Sensitive attribute values.
    c : Series
        Cluster assignment.

    Returns
    -------
    float
        Maximum Kolmogorov-Smirnov statistic.

    Notes
    -----
    The maximum Kolmogorov-Smirnov statistic for the sensitive
    attribute's distributions in the dataset :math:`X` and clusters
    :math:`\mathcal{C}` is computed as

    .. math::
        \max_{C \in \mathcal{C}}\
        \sup_{S \in \mathcal{S}}\
        \big|F_C(S) - F_X(S)\big|

    where :math:`\mathcal{S}` is the set of sensitive attribute values,
    and :math:`F_X` and :math:`F_C` are respectively the sensitive
    attribute's cumulative distribution functions in the dataset
    :math:`X` and cluster :math:`C`.

    References
    ----------
    .. [1] Frank J. Massey Jr. `The Kolmogorov-Smirnov Test for
       Goodness of Fit
       <https://doi.org/10.1080/01621459.1951.10500769>`_. Journal of
       the American Statistical Association. 1951.

    """
    P_X = s.value_counts(normalize=True, sort=False)
    F_X = P_X.sort_index().cumsum()
    cluster_grouped = s.groupby(c, sort=False)
    P_C = cluster_grouped.value_counts(normalize=True, sort=False)
    P_C = P_C.unstack(level=c.name, fill_value=0)
    F_C = P_C.sort_index().cumsum(axis='index')
    # ks stat for a cluster is at most 1 which happens when the cdfs at
    # some point are 0 and 1 for the 2 distributions
    absdiff = F_C.subtract(F_X, axis='index').abs()
    ks_stat = absdiff.max(axis='index')
    return ks_stat.max()


# %% Maximum earth mover's distance -- lower is better
def max_emd(s: Series, c: Series) -> float:
    r"""
    Compute maximum earth mover's distance [1]_.

    Lower is better.

    Parameters
    ----------
    s : Series
        Sensitive attribute values.
    c : Series
        Cluster assignment.

    Returns
    -------
    float
        Maximum earth mover's distance.

    Notes
    -----
    The maximum earth mover's distance for the sensitive attribute's
    distributions in the dataset :math:`X` and clusters
    :math:`\mathcal{C}` is computed as

    .. math::
        \max_{C \in \mathcal{C}}\
        \frac{1}{|\mathcal{S}|}\
        \sum_{S \in \mathcal{S}}\ \big|F_C(S) - F_X(S)\big|

    where :math:`\mathcal{S}` is the set of sensitive attribute values,
    and :math:`F_X` and :math:`F_C` are respectively the sensitive
    attribute's cumulative distribution functions in the dataset
    :math:`X` and cluster :math:`C`.

    References
    ----------
    .. [1] Yossi Rubner, Carlo Tomasi, Leonidas J. Guibas. `A Metric
       for Distributions with Applications to Image Databases
       <https://doi.org/10.1109/ICCV.1998.710701>`_. ICCV 1998.

    """
    P_X = s.value_counts(normalize=True, sort=False)
    F_X = P_X.sort_index().cumsum()
    cluster_grouped = s.groupby(c, sort=False)
    P_C = cluster_grouped.value_counts(normalize=True, sort=False)
    P_C = P_C.unstack(level=c.name, fill_value=0)
    F_C = P_C.sort_index().cumsum(axis='index')
    # emd for a cluster is at most the length of `S` which happens when
    # P_1(min(S)) = 1 and P_2(max(S)) = 1, so the area under P1's cdf
    # will be 1x(len(S)-1), and under P2's cdf will be 0
    absdiff = F_C.subtract(F_X, axis='index').abs()
    # using mean instead of sum makes the emd to be strictly < 1
    emd = absdiff.mean(axis='index')
    return emd.max()


# %% Modified Abraham (2020)'s deviation -- lower is better
def modified_abraham_deviation(s: Series, c: Series, n_clusters: int) -> float:
    r"""
    Compute modified Abraham (2020) [1]_'s deviation.

    Lower is better.

    Parameters
    ----------
    s : Series
        Sensitive attribute values.
    c : Series
        Cluster assignment.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    float
        Modified Abraham (2020)'s deviation.

    Notes
    -----
    Abraham (2020, Equation 22) [1]_'s deviation for the cluster
    assignment :math:`\mathcal{C}` given a single continuous sensitive
    attribute :math:`\mathcal{S}` is computed as

    .. math::
        \sum_{C \in \mathcal{C}}\
        \bigg(\frac{|C|}{|X|}\bigg)^2
        \big(C.\mathcal{S} - X.\mathcal{S}\big)^2

    where :math:`X.\mathcal{S}` and :math:`C.\mathcal{S}` are
    respectively the average values of the sensitive attribute across
    objects in the dataset :math:`X` and cluster :math:`C`. Towards
    having a similar weighting to the :math:`k`-means objective,
    Abraham (2020, §5.4) suggests a multiplier
    :math:`\frac{{|X|}^2}{{|\mathcal{C}|}^2}`. As the :math:`k`-means
    objective in this codebase is divided by :math:`|X|`, we also
    divide this multiplier by :math:`|X|`, thus giving us

    .. math::
        \frac{|X|}{{|\mathcal{C}|}^2}\
        \sum_{C \in \mathcal{C}}\
        \bigg(\frac{|C|}{|X|}\bigg)^2
        \big(C.\mathcal{S} - X.\mathcal{S}\big)^2

    Now, the range of values that :math:`\big(C.\mathcal{S} -
    X.\mathcal{S}\big)` takes is not constrained and hence places no
    upper bound on the above formulation. To address this,
    :math:`x.\mathcal{S}` is first standardised over the dataset to
    have zero mean and unit variance. The modified Abraham (2020)'s
    deviation is thus

    .. math::
        \frac{1}{{|\mathcal{C}|}^2 \times |X|}\
        \sum_{C \in \mathcal{C}}\
        \bigg(\sum_{x \in C} x.\mathcal{R}\bigg)^2

    where :math:`x.\mathcal{R}` corresponds to the value of
    :math:`x.\mathcal{S}` standardised over the dataset :math:`X`.

    References
    ----------
    .. [1] Savitha Sam Abraham, Deepak P, Sowmya S. Sundaram. `Fairness
       in Clustering with Multiple Sensitive Attributes
       <https://doi.org/10.5441/002/edbt.2020.26>`_. EDBT 2020.

    """
    return abraham2020.compute_fairness_loss(s=s, c=c, n_clusters=n_clusters)


# %% Pooling window loss -- lower is better
def pooling_window_loss(s: Series, c: Series, window_size: int) -> float:
    r"""
    Pooling window loss.

    Lower is better.

    Parameters
    ----------
    s : Series
        Sensitive attribute values.
    c : Series
        Cluster assignment.
    window_size : int
        Pooling window size.

    Returns
    -------
    float
        Pooling window loss.

    Notes
    -----
    This is the fairness loss term in fair :math:`k`-means with
    proportional representation along a continuous sensitive attribute.
    The pooling window loss for the cluster assignment
    :math:`\mathcal{C}` is computed as

    .. math::
        \frac{1}{|\mathcal{C}|}\
        \sum_{C \in \mathcal{C}}\
        \sum_{w \in W}\
        \big|h_C(w) - h_X(w)\big|

    where :math:`W` is the set of pooling windows over the sensitive
    attribute's values, and :math:`h_X` and :math:`h_C` are
    respectively the probability distributions after sum pooling over
    these windows in the dataset :math:`X` and cluster :math:`C`. Note
    that as :math:`h_X` and :math:`h_C` are probability distributions,

    .. math::
        \sum_{w \in W} h_X(w) = 1
        \qquad \qquad
        \sum_{w \in W} h_C(w) = 1

    for all :math:`C \in \mathcal{C}`.

    """
    return fairkmeans_prc.compute_fairness_loss(
        s=s, c=c, window_size=window_size
        )


# %% Ziko (2021)'s fairness error -- lower is better
def ziko_fairness_error(s: Series, c: Series) -> float:
    r"""
    Ziko (2021) [1]_'s fairness error.

    Lower is better.

    Parameters
    ----------
    s : Series
        Sensitive attribute values.
    c : Series
        Cluster assignment.

    Returns
    -------
    float
        Ziko (2021)'s fairness error.

    Notes
    -----
    Ziko (2021, § Experiments) [1]_'s fairness error for the cluster
    assignment :math:`\mathcal{C}` is computed as

    .. math::
        \frac{1}{|\mathcal{C}|}\
        \sum_{C \in \mathcal{C}}\
        \mathcal{D}_\text{KL}(P_X || P_C)

    where :math:`\mathcal{D}_\text{KL}` is the KL divergence, and
    :math:`P_X` and :math:`P_C` are respectively the sensitive
    attribute's probability distributions in the dataset :math:`X` and
    some cluster :math:`C`.

    References
    ----------
    .. [1] Imtiaz Masud Ziko, Jing Yuan, Eric Granger, Ismail Ben Ayed.
       `Variational Fair Clustering
       <https://doi.org/10.1609/aaai.v35i12.17336>`_. AAAI 2021.

    """
    P_X = s.value_counts(normalize=True, sort=False)
    cluster_grouped = s.groupby(c, sort=False)
    P_C = cluster_grouped.value_counts(normalize=True, sort=False)
    P_C = P_C.unstack(level=c.name, fill_value=1e-100)
    log_term = -np.log(P_C.divide(P_X, axis='index'))
    return np.nansum(log_term.multiply(P_X, axis='index')) / len(P_C.columns)


# %% END OF FILE
