from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# os.environ['NUMBA_DISABLE_JIT'] = '1'
import numpy as np
from numba import njit
from numba_neighbors.benchmark_utils import run_benchmarks, benchmark
from numba_neighbors import kdtree as kd
from dcbs.core.sample import ifp_sample_and_query_np
import functools

N = 1024
sample_size = 512
D = 3
query_r = 0.2
max_neighbors = 192
leaf_size = 32

np.random.seed(124)
data = np.random.uniform(size=(N, D)).astype(kd.FLOAT_TYPE)
data /= np.linalg.norm(data, axis=-1, keepdims=True)


@benchmark()
def ifp():
    tree = kd.KDTree(data, leaf_size)
    return tree.ifp_sample_query(query_r**2, tree.get_node_indices(),
                                 sample_size, max_neighbors)


@benchmark()
def rejection_ifp():
    tree = kd.KDTree(data, leaf_size)
    return tree.rejection_ifp_sample_query(query_r**2, tree.get_node_indices(),
                                           sample_size, max_neighbors)


@benchmark()
def base():
    return ifp_sample_and_query_np(data, query_r, sample_size, max_neighbors,
                                   max_neighbors)


run_benchmarks(20, 100)
sample_result, query_result = rejection_ifp()
counts = query_result.counts
print(np.min(counts), np.max(counts), np.mean(counts))
