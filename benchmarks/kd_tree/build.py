import numpy as np

import sklearn.neighbors
from numba_neighbors import binary_tree as bt
from numba_neighbors import kd_tree as kd
from numba_neighbors.benchmark_utils import benchmark, run_benchmarks

N = 1024
D = 3

np.random.seed(123)
data = np.random.uniform(size=(N, D)).astype(bt.FLOAT_TYPE)
leaf_size = 16


@benchmark("sklearn")
def sklearn_impl():
    return sklearn.neighbors.KDTree(data, leaf_size=leaf_size)


@benchmark("base")
def numba_impl():
    return bt.create_tree_data(data, leaf_size=leaf_size)


@benchmark("BinaryTree")
def binary_tree():
    return bt.binary_tree(data, leaf_size=leaf_size)


@benchmark("KDTree")
def kd_tree():
    return kd.KDTree(data, leaf_size=leaf_size)


run_benchmarks(10, 100)
