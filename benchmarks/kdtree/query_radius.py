from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numba_neighbors.benchmark_utils import run_benchmarks, benchmark
from numba_neighbors import kdtree as kd
import sklearn.neighbors

N = 1024
n = 256
D = 3
r = 0.1
r2 = r * r
max_neighbors = 32
leaf_size = 16

np.random.seed(123)
data = np.random.uniform(size=(N, D)).astype(kd.FLOAT_TYPE)
X_indices = np.random.choice(N, size=n, replace=False)
X = data[X_indices]

sk_tree = sklearn.neighbors.kd_tree.KDTree(data, leaf_size=leaf_size)


@benchmark('sklearn')
def sklearn_impl():
    return sk_tree.query_radius(X, r, return_distance=True)


numba_data = kd.get_tree_data(data, leaf_size=leaf_size)


@benchmark('numba')
def numba_base():
    dists = np.full((n, max_neighbors), np.inf, dtype=np.float32)
    indices = np.zeros((n, max_neighbors), dtype=np.int64)
    counts = np.zeros((n,), dtype=np.int64)
    kd.query_radius_prealloc(
        X,
        r2,
        dists,
        indices,
        counts,
        n_samples=numba_data.n_samples,
        n_features=numba_data.n_features,
        leaf_size=numba_data.leaf_size,
        n_levels=numba_data.n_levels,
        n_nodes=numba_data.n_nodes,
        data=numba_data.data,
        idx_array=numba_data.idx_array,
        idx_start=numba_data.idx_start,
        idx_end=numba_data.idx_end,
        is_leaf=numba_data.is_leaf,
        node_lower_bounds=numba_data.node_lower_bounds,
        node_upper_bounds=numba_data.node_upper_bounds,
    )


@benchmark('numba_bu')
def numba_bottom_up_impl():
    dists = np.full((n, max_neighbors), np.inf, dtype=np.float32)
    indices = np.zeros((n, max_neighbors), dtype=np.int64)
    counts = np.zeros((n,), dtype=np.int64)
    start_nodes = kd.get_node_indices(
        numba_data.n_samples,
        numba_data.n_nodes,
        numba_data.idx_array,
        numba_data.idx_start,
        numba_data.idx_end,
        numba_data.is_leaf,
    )[X_indices]
    kd.query_radius_bottom_up_prealloc(
        X,
        r2,
        start_nodes,
        dists,
        indices,
        counts,
        n_samples=numba_data.n_samples,
        n_features=numba_data.n_features,
        leaf_size=numba_data.leaf_size,
        n_levels=numba_data.n_levels,
        n_nodes=numba_data.n_nodes,
        data=numba_data.data,
        idx_array=numba_data.idx_array,
        idx_start=numba_data.idx_start,
        idx_end=numba_data.idx_end,
        is_leaf=numba_data.is_leaf,
        node_lower_bounds=numba_data.node_lower_bounds,
        node_upper_bounds=numba_data.node_upper_bounds,
    )


numba_tree = kd.KDTree(data, leaf_size=leaf_size)


@benchmark('numba_cls_pre')
def numba_cls_prealloc():
    dists = np.full((n, max_neighbors), np.inf, dtype=np.float32)
    indices = np.zeros((n, max_neighbors), dtype=np.int64)
    counts = np.zeros((n,), dtype=np.int64)
    numba_tree.query_radius_prealloc(X, r2, dists, indices, counts)


@benchmark()
def numba_cls():
    return numba_tree.query_radius(X, r2, max_neighbors)


@benchmark('numba_cls_bu')
def numba_cls_bottom_up():
    start_nodes = numba_tree.get_node_indices()[X_indices]
    return numba_tree.query_radius_bottom_up(X, r2, start_nodes, max_neighbors)


@benchmark('numba_cls_bu_pre')
def numba_cls_bottom_up_prealloc():
    dists = np.full((n, max_neighbors), np.inf, dtype=np.float32)
    indices = np.zeros((n, max_neighbors), dtype=np.int64)
    counts = np.zeros((n,), dtype=np.int64)
    start_nodes = numba_tree.get_node_indices()[X_indices]
    numba_tree.query_radius_bottom_up_prealloc(X, r2, start_nodes, dists,
                                               indices, counts)


run_benchmarks(20, 1000)
