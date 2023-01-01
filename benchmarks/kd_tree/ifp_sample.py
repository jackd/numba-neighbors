# os.environ['NUMBA_DISABLE_JIT'] = '1'
import numpy as np

# from dcbs.np_utils.ifp.sample_query import ifp_sample_and_query
from numba_neighbors import kd_tree as kd
from numba_neighbors.benchmark_utils import benchmark, run_benchmarks

N = 1024
sample_size = 512
D = 3
rejection_r = 0.2
query_r = 0.2
max_neighbors = 256
leaf_size = 64

np.random.seed(124)
data = np.random.uniform(size=(N, D)).astype(kd.FLOAT_TYPE)
data /= np.linalg.norm(data, axis=-1, keepdims=True)


@benchmark()
def ifp():
    tree = kd.KDTree(data, leaf_size)
    return tree.ifp_sample_query(
        query_r ** 2, tree.get_node_indices(), sample_size, max_neighbors
    )


@benchmark()
def rejection_ifp():
    tree = kd.KDTree(data, leaf_size)
    return tree.rejection_ifp_sample_query(
        rejection_r ** 2,
        query_r ** 2,
        tree.get_node_indices(),
        sample_size,
        max_neighbors,
    )


@benchmark()
def ifp3():
    tree = kd.KDTree3(data, leaf_size)
    return tree.ifp_sample_query(
        query_r ** 2, tree.get_node_indices(), sample_size, max_neighbors
    )


@benchmark()
def rejection_ifp3():
    tree = kd.KDTree3(data, leaf_size)
    return tree.rejection_ifp_sample_query(
        rejection_r ** 2,
        query_r ** 2,
        tree.get_node_indices(),
        sample_size,
        max_neighbors,
    )


# @benchmark()
# def base():
#     return ifp_sample_and_query(
#         data, query_r, sample_size, max_neighbors, max_neighbors
#     )


run_benchmarks(20, 100)
sample_result, query_result = rejection_ifp()
counts = query_result.counts
print(np.min(counts), np.max(counts), np.mean(counts))
