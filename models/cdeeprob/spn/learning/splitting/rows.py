# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala, Federico Luzzi

from typing import Union, Type, Tuple, List, Callable, Any

import numpy as np

from cdeeprob.spn.structure.leaf import Leaf
from cdeeprob.spn.learning.splitting.cluster import gmm, kmeans, kmeans_mb, dbscan, wald
from cdeeprob.spn.learning.splitting.rdc import rdc_rows
from cdeeprob.spn.learning.splitting.random import random_rows

#: A signature for a rows splitting function.
SplitRowsFunc = Callable[
    [np.ndarray,                # The data
     List[Type[Leaf]],          # The distributions
     List[Union[list, tuple]],  # The domains
     np.random.RandomState,     # The random state
     Any],                      # Other arguments
    np.ndarray                  # The rows ids
]


def split_rows_clusters(
    data: np.ndarray,
    clusters: np.ndarray
) -> Tuple[List[np.ndarray], List[list]]:
    """
    Split the data horizontally given the clusters.

    :param data: The data.
    :param clusters: The clusters.
    :return: (slices, weights) where slices is a list of partial data and
             weights is a list of proportions of the local data in respect to the original data.
    """
    slices = list()
    weights = list()
    n_samples = len(data)
    unique_clusters = np.unique(clusters)
    for c in unique_clusters:
        local_data = data[clusters == c, :]
        slices.append(local_data)
        weights.append(len(local_data) / n_samples)
    return slices, weights


def split_rows_clusters_credal(
    data: np.ndarray,
    nans: np.ndarray,
    clusters: np.ndarray
) -> Tuple[List[np.ndarray], List[list]]:
    """
    Split the data horizontally given the clusters.

    :param data: The data.
    :param clusters: The clusters.
    :return: (slices, weights) where slices is a list of partial data and
             weights is a list of proportions of the local data in respect to the original data.
    """
    slices = list()
    maxweights = list()
    minweights = list()
    n_samples = len(data) + len(nans)
    unique_clusters = np.unique(clusters)
    
    if len(unique_clusters) == 1:
        slices.append(data)
        maxweights.append(0)
        minweights.append(0)
        return slices, maxweights, minweights
        
    for c in unique_clusters:
        local_data = data[clusters == c, :]
        lower = len(local_data) / n_samples
        local_data = np.vstack([local_data, nans])
        upper = len(local_data) / n_samples
        slices.append(local_data)
        maxweights.append(upper)
        minweights.append(lower)
    return slices, maxweights, minweights


def get_split_rows_method(split_rows: str) -> SplitRowsFunc:
    """
    Get the rows splitting method given a string.

    :param split_rows: The string of the method do get.
    :return: The corresponding rows splitting function.
    :raises ValueError: If the rows splitting method is unknown.
    """
    if split_rows == 'kmeans':
        return kmeans
    if split_rows == 'kmeans_mb':
        return kmeans_mb
    if split_rows == 'dbscan':
        return dbscan
    if split_rows == 'wald':
        return wald
    if split_rows == 'gmm':
        return gmm
    if split_rows == 'rdc':
        return rdc_rows
    if split_rows == 'random':
        return random_rows
    raise ValueError("Unknown split rows method called {}".format(split_rows))
