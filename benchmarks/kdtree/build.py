from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numba_neighbors.benchmark_utils import run_benchmarks
from numba_neighbors import binary_tree as bt
from numba_neighbors import kdtree as kd
import sklearn.neighbors

N = 1024
D = 3

np.random.seed(123)
data = np.random.uniform(size=(N, D)).astype(bt.FLOAT_TYPE)
leaf_size = 16


def sklearn_impl():
    return sklearn.neighbors.kd_tree.KDTree(data, leaf_size=leaf_size)


def numba_impl():
    return kd.get_tree_data(data, leaf_size=leaf_size)


def numba_class_impl():
    return kd.KDTree(data, leaf_size=leaf_size)


run_benchmarks(
    10,
    100,
    ('sklearn', sklearn_impl),
    ('numba', numba_impl),
    ('numba_class', numba_class_impl),
)
