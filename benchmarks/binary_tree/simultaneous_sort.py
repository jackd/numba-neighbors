from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numba_neighbors.benchmark_utils import run_benchmarks
from numba_neighbors import binary_tree as bt
import sklearn.neighbors

N = 2048
k = 32

np.random.seed(123)
dst = np.random.uniform(size=(N, k))
idx = np.random.uniform(size=(N, k), high=N).astype(np.int64)


def get_data():
    return dst.copy(), idx.copy()


def sklearn_impl():
    data = get_data()
    sklearn.neighbors.kd_tree.simultaneous_sort(*data)
    return data


def numba_impl():
    data = get_data()
    bt.simultaneous_sort(*data)
    return data


def numpy_sort(dst, idx):
    i = np.argsort(dst)
    expected_dst = np.take_along_axis(dst, i, axis=1)
    expected_idx = np.take_along_axis(idx, i, axis=1)
    return expected_dst, expected_idx


def numpy_impl():
    dst, idx = get_data()
    return numpy_sort(dst, idx)


run_benchmarks(
    10,
    100,
    ('sklearn', sklearn_impl),
    ('numba', numba_impl),
    ('numpy', numpy_impl),
)
