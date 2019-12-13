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
from numba_neighbors import binary_tree as bt
from numba_neighbors import kd_tree as kd
from dcbs.sparse.sample import ragged_in_place_and_down_sample_query_np
import functools
import heapq
heapq.heappush

n0 = 1024
n1 = 256
r0 = 0.1
r1 = 0.2
max_neigh0 = 64
max_neigh1 = 256
# n0 = 256
# n1 = 64
# r0 = 0.2
# r1 = 0.2 * np.sqrt(2)
# max_neigh0 = 64
# max_neigh1 = 128

D = 3
leaf_size = 32

np.random.seed(124)

data = np.random.uniform(size=(n0, D)).astype(kd.FLOAT_TYPE)
data /= np.linalg.norm(data, axis=-1, keepdims=True)

kwargs = dict(data=data,
              leaf_size=leaf_size,
              n1=n1,
              r0=r0,
              r1=r1,
              max_neigh0=max_neigh0,
              max_neigh1=max_neigh1)


# @benchmark('numba')
@njit()
def numba_impl(data, leaf_size, n1, r0, r1, max_neigh0, max_neigh1):
    tree = kd.KDTree(data, leaf_size)

    start_nodes = tree.get_node_indices()
    query0 = tree.query_radius_bottom_up(data, r0**2, start_nodes, max_neigh0)

    sample = bt.rejection_sample_precomputed(query0.indices, query0.counts, n1,
                                             n0, tree.int_type, tree.bool_type)
    sample_indices = sample.indices[:sample.count]
    query1 = tree.query_radius_bottom_up(data[sample_indices], r1**2,
                                         start_nodes[sample_indices],
                                         max_neigh1)

    return query0, sample, query1


num_impl = benchmark('numba')(functools.partial(numba_impl, **kwargs))


@njit()
def numba3_impl(data, leaf_size, n1, r0, r1, max_neigh0, max_neigh1):
    tree = kd.KDTree3(data, leaf_size)

    start_nodes = tree.get_node_indices()
    query0 = tree.query_radius_bottom_up(data, r0**2, start_nodes, max_neigh0)

    sample = bt.rejection_sample_precomputed(query0.indices, query0.counts, n1,
                                             n0, tree.int_type, tree.bool_type)
    sample_indices = sample.indices[:sample.count]
    query1 = tree.query_radius_bottom_up(data[sample_indices], r1**2,
                                         start_nodes[sample_indices],
                                         max_neigh1)
    return query0, sample, query1


benchmark('numba3')(functools.partial(numba3_impl, **kwargs))


@benchmark()
def original():
    (ip_dists, ip_indices, sample_indices, sample_size, ds_dists,
     ds_indices) = ragged_in_place_and_down_sample_query_np(
         data, n0, r0, max_neigh0, n1, r1, max_neigh1)
    q0 = bt.QueryResult(ip_dists, ip_indices,
                        np.count_nonzero(np.isfinite(ip_dists), axis=1))
    sample = bt.RejectionSampleResult(sample_indices, sample_size)
    q1 = bt.QueryResult(ds_dists, ds_indices,
                        np.count_nonzero(np.isfinite(ds_dists), axis=1))
    return q0, sample, q1


run_benchmarks(20, 100)
q0, s, q1 = num_impl()

print(np.max(q0.counts), np.mean(q0.counts))
print(s.count)
counts = q1.counts[:s.count]
print(np.max(counts), np.mean(counts))

orig_q0, orig_s, orig_q1 = original()
print('---')
print(np.max(orig_q0.counts), np.mean(orig_q0.counts))
print(orig_s.count)
counts = orig_q1.counts[:orig_s.count]
print(np.max(counts), np.mean(counts))
print(orig_q1.dists.shape)
