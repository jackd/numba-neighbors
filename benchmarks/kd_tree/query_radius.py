import numpy as np
import sklearn.neighbors

from numba_neighbors import kd_tree as kd
from numba_neighbors.benchmark_utils import benchmark, run_benchmarks

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


@benchmark("sklearn")
def sklearn_impl():
    return sk_tree.query_radius(X, r, return_distance=True)


numba_tree = kd.KDTree(data, leaf_size=leaf_size)


@benchmark("numba_pre")
def numba_prealloc():
    dists = np.full((n, max_neighbors), np.inf, dtype=np.float32)
    indices = np.zeros((n, max_neighbors), dtype=np.int64)
    counts = np.zeros((n,), dtype=np.int64)
    numba_tree.query_radius_prealloc(X, r2, dists, indices, counts)


@benchmark()
def numba():
    return numba_tree.query_radius(X, r2, max_neighbors)


@benchmark("numba_bu")
def numba_bottom_up():
    start_nodes = numba_tree.get_node_indices()[X_indices]
    return numba_tree.query_radius_bottom_up(X, r2, start_nodes, max_neighbors)


@benchmark("numba_bu_pre")
def numba_bottom_up_prealloc():
    dists = np.full((n, max_neighbors), np.inf, dtype=np.float32)
    indices = np.zeros((n, max_neighbors), dtype=np.int64)
    counts = np.zeros((n,), dtype=np.int64)
    start_nodes = numba_tree.get_node_indices()[X_indices]
    numba_tree.query_radius_bottom_up_prealloc(
        X, r2, start_nodes, dists, indices, counts
    )


numba_tree3 = kd.KDTree3(data, leaf_size=leaf_size)


@benchmark()
def numba3():
    return numba_tree3.query_radius(X, r2, max_neighbors)


@benchmark("numba3_bu")
def numba3_bottom_up():
    start_nodes = numba_tree3.get_node_indices()[X_indices]
    return numba_tree3.query_radius_bottom_up(X, r2, start_nodes, max_neighbors)


@benchmark("numba3_bu_pre")
def numba3_bottom_up_prealloc():
    dists = np.full((n, max_neighbors), np.inf, dtype=np.float32)
    indices = np.zeros((n, max_neighbors), dtype=np.int64)
    counts = np.zeros((n,), dtype=np.int64)
    start_nodes = numba_tree3.get_node_indices()[X_indices]
    numba_tree3.query_radius_bottom_up_prealloc(
        X, r2, start_nodes, dists, indices, counts
    )


# numba_tree2 = kd2.KDTree(data, leaf_size=leaf_size)

# @benchmark()
# def numba2_cls():
#     return numba_tree2.query_radius(X, r2, max_neighbors)

# @benchmark('numba2_cls_bu')
# def numba2_cls_bottom_up():
#     start_nodes = numba_tree2.get_node_indices()[X_indices]
#     return numba_tree2.query_radius_bottom_up(X, r2, start_nodes, max_neighbors)

# @benchmark('numba2_cls_bu_pre')
# def numba2_cls_bottom_up_prealloc():
#     dists = np.full((n, max_neighbors), np.inf, dtype=np.float32)
#     indices = np.zeros((n, max_neighbors), dtype=np.int64)
#     counts = np.zeros((n,), dtype=np.int64)
#     start_nodes = numba_tree2.get_node_indices()[X_indices]
#     numba_tree2.query_radius_bottom_up_prealloc(X, r2, start_nodes, dists,
#                                                 indices, counts)

run_benchmarks(20, 1000)
