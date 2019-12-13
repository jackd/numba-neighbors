from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numba_neighbors.benchmark_utils import run_benchmarks, benchmark
from numba_neighbors import binary_tree as bt
from numba_neighbors import kd_tree as kd
# from numba_neighbors import kd_tree2 as kd2
import sklearn.neighbors

N = 1024
D = 3

np.random.seed(123)
data = np.random.uniform(size=(N, D)).astype(bt.FLOAT_TYPE)
leaf_size = 16


@benchmark('sklearn')
def sklearn_impl():
    return sklearn.neighbors.kd_tree.KDTree(data, leaf_size=leaf_size)


@benchmark('base')
def numba_impl():
    return bt.get_tree_data(data, leaf_size=leaf_size)


@benchmark('BinaryTree')
def binary_tree():
    # return kd.KDTree(data, leaf_size=leaf_size)
    return bt.BinaryTree(data, leaf_size=leaf_size)


@benchmark('KDTree')
def kd_tree():
    # return kd.KDTree(data, leaf_size=leaf_size)
    return kd.KDTree(data, leaf_size=leaf_size)


# @benchmark('cls2')
# def numba2():
#     return kd2.KDTree(data, leaf_size=leaf_size)

run_benchmarks(10, 100)
