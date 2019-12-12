from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numba_neighbors.benchmark_utils import run_benchmarks
from numba_neighbors import binary_tree as bt
import sklearn.neighbors

N = 2048
NS = 1024
D = 3

np.random.seed(123)
data = np.random.uniform(size=(N, D))
node_indices = np.random.uniform(size=(NS,), high=N).astype(np.int64)


def get_data():
    return data.copy(), node_indices.copy()


# def sklearn_impl():
#     data, node_indices = get_data()
#     n_points = node_indices.size
#     n_features = data.shape[1]
#     return sklearn.neighbors.kd_tree.find_node_split_dim(
#         data, node_indices, n_features, n_points)


def numba_impl():
    data = get_data()
    return bt.find_node_split_dim(*data)


def numpy_find(data, node_indices):
    data = data[node_indices]
    return np.argmax(np.ptp(data))


def numpy_impl():
    return numpy_find(*get_data())


run_benchmarks(
    10,
    100,
    # ('sklearn', sklearn_impl),
    ('numba', numba_impl),
    ('numpy', numpy_impl),
)
