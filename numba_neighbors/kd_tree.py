import numba as nb
import numpy as np
from numba.experimental import jitclass

from numba_neighbors import binary_tree as bt
from numba_neighbors.binary_tree import (
    BOOL_TYPE,
    FASTMATH,
    FLOAT_TYPE,
    INT_TYPE,
    PARALLEL,
    FloatArray,
)

# @nb.njit(parallel=PARALLEL, fastmath=FASTMATH)
# def _update_node_radii(n_features, n_nodes, radius, node_lower_bounds,
#                        node_upper_bounds):

#     for i_node in nb.prange(n_nodes):
#         rad = 0
#         ub = node_upper_bounds[i_node]
#         lb = node_lower_bounds[i_node]
#         for j in range(n_features):
#             # if upper_bounds[j] != np.inf:
#             next_val = 0.5 * abs(ub[j] - lb[j])
#             rad += next_val * next_val

#         # The radius will hold the size of the circumscribed hypersphere measured
#         # with the specified metric: in querying, this is used as a measure of the
#         # size of each node when deciding which nodes to split.
#         radius[i_node] = rad


def create_kd_node_data_nojit(n_nodes, n_features, float_type=FLOAT_TYPE):
    return np.empty((n_nodes, 2, n_features), dtype=float_type)


create_kd_node_data = nb.njit()(create_kd_node_data_nojit)


def create_kd_tree_data_nojit(
    data: FloatArray, leaf_size: int = 40, int_type=INT_TYPE, bool_type=BOOL_TYPE
):

    idx_array, idx_start, idx_end, is_leaf = bt.create_tree_data.py_func(
        data, leaf_size, int_type, bool_type
    )
    n_features = data.shape[1]
    n_nodes = idx_start.size
    node_data = create_kd_node_data_nojit(n_nodes, n_features, data.dtype)
    return (idx_array, idx_start, idx_end, is_leaf, node_data)


@nb.njit()
def create_kd_tree_data(
    data: FloatArray, leaf_size: int = 40, int_type=INT_TYPE, bool_type=BOOL_TYPE
):

    idx_array, idx_start, idx_end, is_leaf = bt.create_tree_data(
        data, leaf_size, int_type, bool_type
    )
    n_features = data.shape[1]
    n_nodes = idx_start.size
    node_data = create_kd_node_data(n_nodes, n_features, data.dtype)
    return (idx_array, idx_start, idx_end, is_leaf, node_data)


@nb.njit(parallel=PARALLEL, fastmath=FASTMATH)
def fill_kd_tree_node_data(
    n_features, n_nodes, data, idx_array, idx_start, idx_end, float_type, node_data
):
    """Initialize the node for the dataset stored in self.data"""
    # node_bounds = np.empty((n_nodes, 2, n_features), dtype=float_type)

    node_data[:, 0] = np.inf
    node_data[:, 1] = -np.inf

    for i_node in nb.prange(n_nodes):  # pylint: disable=not-an-iterable
        idx_start_value = idx_start[i_node]
        idx_end_value = idx_end[i_node]

        lower_bounds = node_data[i_node, 0]
        upper_bounds = node_data[i_node, 1]

        # Compute the actual data range.  At build time, this is slightly
        # slower than using the previously-computed bounds of the parent node,
        # but leads to more compact trees and thus faster queries.

        for i in range(idx_start_value, idx_end_value):
            data_row = data[idx_array[i]]
            for j in range(n_features):
                val = data_row[j]
                lower_bounds[j] = min(lower_bounds[j], val)
                upper_bounds[j] = max(upper_bounds[j], val)


@nb.njit(inline="always", fastmath=FASTMATH)
def min_max_rdist(node_data, x: FloatArray):
    """Compute the minimum and maximum distance between a point and a node"""
    lower_bounds = node_data[0]
    upper_bounds = node_data[1]
    min_dist = 0.0
    max_dist = 0.0

    for j in range(x.size):
        d_lo = lower_bounds[j] - x[j]
        d_hi = x[j] - upper_bounds[j]
        # d = ((d_lo + abs(d_lo)) + (d_hi + abs(d_hi)))  # twice as big as actual
        d = max(d_lo, 0) + max(d_hi, 0)
        min_dist += d * d
        d = max(abs(d_lo), abs(d_hi))
        max_dist += d * d
    # min_dist *= 0.25
    return min_dist, max_dist


@nb.njit(inline="always", fastmath=FASTMATH)
def rdist(x, y):
    acc = 0
    for i in range(x.size):
        diff = x[i] - y[i]
        # acc += pow(diff, 2)
        acc += diff * diff
    return acc


@nb.njit(inline="always", fastmath=FASTMATH)
def min_max_rdist3(node_data: FloatArray, x: FloatArray):
    """min_max_rdist implementation optimized for 3D arrays."""
    lower_bounds = node_data[0]
    upper_bounds = node_data[1]

    d_lo0 = lower_bounds[0] - x[0]
    d_hi0 = x[0] - upper_bounds[0]

    d_lo1 = lower_bounds[1] - x[1]
    d_hi1 = x[1] - upper_bounds[1]

    d_lo2 = lower_bounds[2] - x[2]
    d_hi2 = x[2] - upper_bounds[2]

    min_dist = (
        (max(d_lo0, 0) + max(d_hi0, 0)) ** 2
        + (max(d_lo1, 0) + max(d_hi1, 0)) ** 2
        + (max(d_lo2, 0) + max(d_hi2, 0)) ** 2
    )

    max_dist = (
        (max(abs(d_lo0), abs(d_hi0))) ** 2
        + (max(abs(d_lo1), abs(d_hi1))) ** 2
        + (max(abs(d_lo2), abs(d_hi2))) ** 2
    )

    return min_dist, max_dist


@nb.njit(inline="always", fastmath=FASTMATH)
def rdist3(x, y):
    """Optimized rdist for 3D arrays."""
    return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2


class KDTreeBase(bt.BinaryTree):
    def _fill_node_data(self):
        fill_kd_tree_node_data(
            self.n_features,
            self.n_nodes,
            self.data,
            self.idx_array,
            self.idx_start,
            self.idx_end,
            self.float_type,
            self.node_data,
        )


def kdtree_spec(float_type=FLOAT_TYPE, int_type=INT_TYPE, bool_type=BOOL_TYPE):
    float_type_t = nb.from_dtype(float_type)
    return [
        *bt.tree_spec(float_type, int_type, bool_type),
        ("node_data", float_type_t[:, :, ::1]),
    ]


@jitclass(kdtree_spec())
class _KDTree(KDTreeBase):
    @property
    def float_type(self):
        return FLOAT_TYPE

    @property
    def int_type(self):
        return INT_TYPE

    @property
    def bool_type(self):
        return BOOL_TYPE

    @property
    def rdist(self):
        return rdist

    @property
    def min_max_rdist(self):
        return min_max_rdist


@nb.njit()
def KDTree(data: FloatArray, leaf_size: int = 40):
    (idx_array, idx_start, idx_end, is_leaf, node_data) = create_kd_tree_data(
        data, leaf_size
    )
    return _KDTree(data, leaf_size, idx_array, idx_start, idx_end, is_leaf, node_data)


def KDTree_nojit(data: FloatArray, leaf_size: int = 40):
    (idx_array, idx_start, idx_end, is_leaf, node_data) = create_kd_tree_data_nojit(
        data, leaf_size
    )
    return _KDTree(data, leaf_size, idx_array, idx_start, idx_end, is_leaf, node_data)


@jitclass(kdtree_spec())
class _KDTree3(KDTreeBase):
    @property
    def float_type(self):
        return FLOAT_TYPE

    @property
    def int_type(self):
        return INT_TYPE

    @property
    def bool_type(self):
        return BOOL_TYPE

    @property
    def rdist(self):
        return rdist3

    @property
    def min_max_rdist(self):
        return min_max_rdist3


@nb.njit()
def KDTree3(data: FloatArray, leaf_size: int = 40):
    (idx_array, idx_start, idx_end, is_leaf, node_data) = create_kd_tree_data(
        data, leaf_size
    )
    return _KDTree3(data, leaf_size, idx_array, idx_start, idx_end, is_leaf, node_data)


def KDTree3_nojit(data: FloatArray, leaf_size: int = 40):
    (idx_array, idx_start, idx_end, is_leaf, node_data) = create_kd_tree_data_nojit(
        data, leaf_size
    )
    return _KDTree3(data, leaf_size, idx_array, idx_start, idx_end, is_leaf, node_data)
