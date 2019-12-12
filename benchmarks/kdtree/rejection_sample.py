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
from dcbs.sparse.sample import ragged_in_place_and_down_sample_query_np
import functools

N = 1024
max_sample_size = 512
D = 3
rejection_r = 0.1
query_r = 0.2
max_neighbors = 256
leaf_size = 32

np.random.seed(124)
data = np.random.uniform(size=(N, D)).astype(kd.FLOAT_TYPE)
data /= np.linalg.norm(data, axis=-1, keepdims=True)


@njit()
def numba_impl(data, leaf_size):
    tree = kd.KDTree(data, leaf_size)
    return tree.rejection_sample_query(rejection_r**2, query_r**2,
                                       tree.get_node_indices(), max_sample_size,
                                       max_neighbors)


@benchmark()
def separate():
    ragged_in_place_and_down_sample_query_np(data, N, rejection_r,
                                             max_neighbors, max_sample_size,
                                             query_r, max_neighbors)


sample_result, query_result = numba_impl(data, leaf_size)
benchmark('numba_impl')(functools.partial(numba_impl,
                                          data=data,
                                          leaf_size=leaf_size))
run_benchmarks(20, 100)

count = sample_result.count
print(count)
counts = query_result.counts[:count]
print(np.min(counts), np.max(counts), np.mean(counts))
