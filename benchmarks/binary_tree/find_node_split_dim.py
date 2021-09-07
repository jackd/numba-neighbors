import numpy as np

from numba_neighbors import binary_tree as bt
from numba_neighbors.benchmark_utils import benchmark, run_benchmarks

N = 2048
NS = 1024
D = 3

np.random.seed(123)
data = np.random.uniform(size=(N, D))
node_indices = np.random.uniform(size=(NS,), high=N).astype(np.int64)


def get_data():
    return data.copy(), node_indices.copy()


# @benchmark('sklearn')
# def sklearn_impl():
#     data, node_indices = get_data()
#     n_points = node_indices.size
#     n_features = data.shape[1]
#     return sklearn.neighbors._kd_tree.find_node_split_dim(
#         data, node_indices, n_features, n_points)


@benchmark("numba")
def numba_impl():
    data = get_data()
    return bt.find_node_split_dim(*data)


def numpy_find(data, node_indices):
    data = data[node_indices]
    return np.argmax(np.ptp(data))


@benchmark("numpy")
def numpy_impl():
    return numpy_find(*get_data())


run_benchmarks(10, 100)
