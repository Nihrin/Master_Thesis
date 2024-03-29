# MIT License: Copyright (c) 2021 Lorenzo Loconte, Gennaro Gala, Federico Luzzi

from typing import Union, Type, List
from collections import deque

import numpy as np

from cdeeprob.spn.structure.leaf import LeafType, Leaf


def gvs_cols(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    p: float = 5.0
) -> np.ndarray:
    """
    Greedy Variable Splitting (GVS) independence test.

    :param data: The data.
    :param distributions: The distributions.
    :param domains: The domains.
    :param random_state: The random state.
    :param p: The threshold for the G-Test.
    :return: A partitioning of features.
    :raises ValueError: If the leaf distributions are discrete and continuous.
    """
    _, n_features = data.shape
    rand_init = random_state.randint(0, n_features)
    features_set = set(filter(lambda x: x != rand_init, range(n_features)))
    dependent_features_set = {rand_init}

    features_queue = deque()
    features_queue.append(rand_init)

    while features_queue:
        feature = features_queue.popleft()
        features_remove = set()
        for other_feature in features_set:
            if not gtest(data, feature, other_feature, distributions, domains, p=p):
                features_remove.add(other_feature)
                dependent_features_set.add(other_feature)
                features_queue.append(other_feature)
        features_set = features_set.difference(features_remove)

    partition = np.zeros(n_features, dtype=np.int64)
    partition[list(dependent_features_set)] = 1
    return partition


def rgvs_cols(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    p: float = 5.0
) -> np.ndarray:
    """
    Random Greedy Variable Splitting (RGVS) independence test.

    :param data: The data.
    :param distributions: The distributions.
    :param domains: The domains.
    :param random_state: The random state.
    :param p: The threshold for the G-Test.
    :return: A partitioning of features.
    :raises ValueError: If the leaf distributions are discrete and continuous.
    """
    _, n_features = data.shape
    k = int(max(np.sqrt(n_features), 2))
    if k == n_features:
        return gvs_cols(data, distributions, domains, random_state, p)

    rand_perm = random_state.permutation(np.arange(n_features))[:k]
    data_gvs = data[:, rand_perm]
    distributions_gvs = []
    domains_gvs = []
    for e in rand_perm:
        distributions_gvs.append(distributions[e])
        domains_gvs.append(domains[e])

    partition_gvs = gvs_cols(data_gvs, distributions_gvs, domains_gvs, random_state, p)

    if (partition_gvs != 1).all():
        return np.zeros(n_features, dtype=np.int64)

    if random_state.rand() < 0.5:
        # excluded in first cluster 0
        partition = np.zeros(n_features, dtype=np.int64)
    else:
        # excluded in second cluster 1
        partition = np.ones(n_features, dtype=np.int64)

    partition[rand_perm] = partition_gvs
    return partition


def wrgvs_cols(
    data: np.ndarray,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    random_state: np.random.RandomState,
    p: float = 5.0
) -> np.ndarray:
    """
    Wiser Random Greedy Variable Splitting (WRGVS) independence test.

    :param data: The data.
    :param distributions: The distributions.
    :param domains: The domains.
    :param random_state: The random state.
    :param p: The threshold for the G-Test.
    :return: A partitioning of features.
    :raises ValueError: If the leaf distributions are discrete and continuous.
    """
    _, n_features = data.shape
    k = int(max(np.sqrt(n_features), 2))
    if k == n_features:
        return gvs_cols(data, distributions, domains, random_state, p)

    rand_perm = random_state.permutation(np.arange(n_features))[:k]
    data_gvs = data[:, rand_perm]
    distributions_gvs = []
    domains_gvs = []
    for e in rand_perm:
        distributions_gvs.append(distributions[e])
        domains_gvs.append(domains[e])

    partition_gvs = gvs_cols(data_gvs, distributions_gvs, domains_gvs, random_state, p)

    if ((partition_gvs != 1).all()) or ((partition_gvs != 0).all()):
        return np.zeros(n_features, dtype=np.int64)

    part_0 = set(rand_perm[partition_gvs == 0])
    part_1 = set(rand_perm[partition_gvs == 1])
    part_0_el = random_state.choice(list(part_0), replace=False)
    part_1_el = random_state.choice(list(part_1), replace=False)
    feature_excluded = set(range(n_features)) - set(rand_perm)

    for f_i in feature_excluded:
        # g testing for deciding which cluster
        g_val0 = gtest(data, f_i, part_0_el, distributions, domains, p, test=False)
        g_val1 = gtest(data, f_i, part_1_el, distributions, domains, p, test=False)
        if g_val0 > g_val1:
            part_0.add(f_i)
        else:
            part_1.add(f_i)

    partition = np.zeros(n_features, dtype=np.int64)
    partition[list(part_1)] = 1
    return partition


def gtest(
    data: np.ndarray,
    i: int,
    j: int,
    distributions: List[Type[Leaf]],
    domains: List[Union[list, tuple]],
    p: float = 5.0,
    test: bool = True
) -> Union[bool, float]:
    """
    The G-Test independence test between two features.

    :param data: The data.
    :param i: The index of the first feature.
    :param j: The index of the second feature.
    :param distributions: The distributions.
    :param domains: The domains.
    :param p: The threshold for the G-Test.
    :param test: If the method is called as test (true) or as value of statistics (false), default True.
    :return: False if the features are assumed to be dependent, True otherwise.
    :raises ValueError: If the leaf distributions are discrete and continuous.
    """
    n_samples = len(data)
    x1, x2 = data[:, i], data[:, j]

    indices1 = np.where(np.isnan(x1))[0]
    indices2 = np.where(np.isnan(x2))[0]
    intersect = np.intersect1d(indices1, indices2)

    if np.any(intersect):
        x1 = np.delete(x1, intersect)
        x2 = np.delete(x2, intersect)
        indices1 = np.where(np.isnan(x1))[0]
        indices2 = np.where(np.isnan(x2))[0]

    indices = np.union1d(indices1, indices2)
    lowx1 = np.delete(x1, indices)
    lowx2 = np.delete(x2, indices)
    x1_ = np.delete(x1, indices1)
    x2_ = np.delete(x2, indices2)

    if distributions[i].LEAF_TYPE == LeafType.DISCRETE:
        b1 = domains[i] + [domains[i][-1] + (domains[i][-1] - domains[i][-2])]
        histx1, _ = np.histogram(x1_, bins=b1)
    elif distributions[i].LEAF_TYPE == LeafType.CONTINUOUS:
        histx1, b1 = np.histogram(x1_, bins='scott')     
    else:
        raise ValueError("Leaf distribution must be either discrete or continuous")
    
    if distributions[j].LEAF_TYPE == LeafType.DISCRETE:
        b2 = domains[j] + [domains[j][-1] + (domains[j][-1] - domains[j][-2])]
        histx2, _ = np.histogram(x2_, bins=b2)
    elif distributions[j].LEAF_TYPE == LeafType.CONTINUOUS:
        histx2, b2 = np.histogram(x2_, bins='scott')
    else:
        raise ValueError("Leaf distribution must be either discrete or continuous")
    
    lowhist, _, _ = np.histogram2d(lowx1, lowx2, bins=[b1, b2])

    
    uphist = lowhist.copy()
    mr = np.sum(lowhist, axis=1, keepdims=True)
    mc = np.sum(lowhist, axis=0, keepdims=True)
    for r in range(len(lowhist)):
        for c in range(len(lowhist[r])):
            diffr = histx1[r] - mr[r][0]
            diffc = histx2[c] - mc[0][c]
            uphist[r,c] = lowhist[r,c] + diffr + diffc

    hist = np.mean(np.array([lowhist, uphist]), axis=0)
    # Compute G-test statistics
    hist = hist.astype(np.float32) + np.finfo(np.float32).eps
    m1 = np.sum(hist, axis=1, keepdims=True)
    m2 = np.sum(hist, axis=0, keepdims=True)
    e = m1 * m2 / n_samples
    g_val = 2.0 * np.sum(hist * np.log(hist / e))

    # Return test result
    if test:
        f1 = np.count_nonzero(m1)
        f2 = np.count_nonzero(m2)
        dof = (f1 - 1) * (f2 - 1)
        p_thresh = 2.0 * dof * p
        return g_val < p_thresh

    # Return the value of G-test
    return g_val
