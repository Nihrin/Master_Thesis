# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala, Federico Luzzi

from typing import Union, Type, List

import numpy as np

from cdeeprob.spn.structure.leaf import Leaf, LeafType
from cdeeprob.utils.statistics import compute_gini


def gini_cols(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    e: float = 0.3,
    alpha: float = 0.1
) -> np.ndarray:
    """
    Gini index column splitting method.

    :param data: The data.
    :param distributions: Distributions of the features.
    :param domains: Range of values of the features.
    :param e: Threshold of the considered entropy to be signficant.
    :param alpha: laplacian alpha to apply at frequence.
    :return: A partitioning of features.
    """
    _, n_features = data.shape
    partition = np.zeros(n_features, dtype=np.int64)

    # Compute Gini index for each variable
    for i in range(n_features):
        if distributions[i].LEAF_TYPE == LeafType.DISCRETE:
            bins = domains[i] + [len(domains[i])]
            hist, _ = np.histogram(data[:, i], bins=bins)
            probs = (hist + alpha) / (len(data) + len(hist) * alpha)
        elif distributions[i].LEAF_TYPE == LeafType.CONTINUOUS:
            hist, _ = np.histogram(data[:, i], bins='scott')
            probs = (hist + alpha) / (len(data) + len(hist) * alpha)
        else:
            raise ValueError("Leaves distributions must be either discrete or continuous")

        # Compute the Gini index
        gini = compute_gini(probs)

        # Add to cluster if the Gini index is less than the threshold
        if gini < e:
            partition[i] = 1

    return partition


def gini_adaptive_cols(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    e: float = 0.3,
    alpha: float = 0.1,
    size: int = None
) -> np.ndarray:
    """
    Adaptive Gini index column splitting method.

    :param data: The data.
    :param distributions: Distributions of the features.
    :param domains: Range of values of the features.
    :param e: Threshold of the considered entropy to be signficant.
    :param alpha: laplacian alpha to apply at frequence.
    :param size: Size of whole dataset.
    :return: A partitioning of features.
    :raises ValueError: If the size of the data is missing.
    """
    if size is None:
        raise ValueError("Missing data size for Adaptive Gini column splitting method")

    return gini_cols(
        data, distributions, domains, random_state,
        e=max(e * (len(data) / size), np.finfo(np.float32).eps),
        alpha=alpha
    )
