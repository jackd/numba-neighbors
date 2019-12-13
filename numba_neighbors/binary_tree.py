from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import Any, Callable, NamedTuple, Tuple, TypeVar
import numba as nb
import numpy as np
from numba_neighbors import index_heap as ih

NodeData = TypeVar('NodeData')

FASTMATH = True

INT_TYPE = np.int64
INT_TYPE_T = nb.int64

FLOAT_TYPE = np.float32
FLOAT_TYPE_T = nb.float32

BOOL_TYPE = np.uint8
BOOL_TYPE_T = nb.uint8

IntArray = np.ndarray
FloatArray = np.ndarray
BoolArray = np.ndarray

RDist = Callable[[FloatArray, FloatArray], float]
MinMaxRDist = Callable[[NodeData, FloatArray], Tuple[float, float]]


@nb.njit(inline='always')
def swap(arr, i1, i2):
    tmp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = tmp


@nb.njit(inline='always')
def dual_swap(darr, iarr, i1, i2):
    """swap the values at inex i1 and i2 of both darr and iarr"""
    dtmp = darr[i1]
    darr[i1] = darr[i2]
    darr[i2] = dtmp

    itmp = iarr[i1]
    iarr[i1] = iarr[i2]
    iarr[i2] = itmp


@nb.njit()
def _simultaneous_sort(dist: FloatArray, idx: IntArray):
    """
    Perform a recursive quicksort on the dist array, simultaneously
    performing the same swaps on the idx array.  The equivalent in
    numpy (though quite a bit slower) is
    def simultaneous_sort(dist, idx):
        i = np.argsort(dist)
        return dist[i], idx[i]
    """
    # in the small-array case, do things efficiently
    size = dist.size
    if size <= 1:
        pass
    elif size == 2:
        if dist[0] > dist[1]:
            dual_swap(dist, idx, 0, 1)
    elif size == 3:
        if dist[0] > dist[1]:
            dual_swap(dist, idx, 0, 1)
        if dist[1] > dist[2]:
            dual_swap(dist, idx, 1, 2)
            if dist[0] > dist[1]:
                dual_swap(dist, idx, 0, 1)
    else:
        # Determine the pivot using the median-of-three rule.
        # The smallest of the three is moved to the beginning of the array,
        # the middle (the pivot value) is moved to the end, and the largest
        # is moved to the pivot index.
        pivot_idx = size // 2
        if dist[0] > dist[size - 1]:
            dual_swap(dist, idx, 0, size - 1)
        if dist[size - 1] > dist[pivot_idx]:
            dual_swap(dist, idx, size - 1, pivot_idx)
            if dist[0] > dist[size - 1]:
                dual_swap(dist, idx, 0, size - 1)
        pivot_val = dist[size - 1]

        # partition indices about pivot.  At the end of this operation,
        # pivot_idx will contain the pivot value, everything to the left
        # will be smaller, and everything to the right will be larger.
        store_idx = 0
        for i in range(size - 1):
            if dist[i] < pivot_val:
                dual_swap(dist, idx, i, store_idx)
                store_idx += 1
        dual_swap(dist, idx, store_idx, size - 1)
        pivot_idx = store_idx

        # recursively sort each side of the pivot
        if pivot_idx > 1:
            _simultaneous_sort(dist[:pivot_idx], idx[:pivot_idx])

        if pivot_idx + 2 < size:
            start = pivot_idx + 1
            _simultaneous_sort(dist[start:], idx[start:])
    return 0


@nb.njit(parallel=True)
def simultaneous_sort(distances: FloatArray, indices: IntArray):
    """In-place simultaneous sort the given row of the arrays

    This python wrapper exists primarily to enable unit testing
    of the _simultaneous_sort C routine.
    """
    assert (distances.shape == indices.shape)
    assert (len(distances.shape) == 2)
    for row in nb.prange(distances.shape[0]):  # pylint: disable=not-an-iterable
        _simultaneous_sort(distances[row], indices[row])


@nb.njit(parallel=True)
def simultaneous_sort_partial(distances: FloatArray, indices: IntArray,
                              counts: IntArray):
    """In-place simultaneous sort the given row of the arrays

    This python wrapper exists primarily to enable unit testing
    of the _simultaneous_sort C routine.
    """
    assert (distances.shape == indices.shape)
    assert (len(distances.shape) == 2)
    assert (distances.shape[:1] == counts.shape)
    for row in nb.prange(distances.shape[0]):  # pylint: disable=not-an-iterable
        count = counts[row]
        _simultaneous_sort(distances[row, :count], indices[row, :count])


@nb.njit()
def find_node_split_dim(data: FloatArray, node_indices: IntArray):
    """Find the dimension with the largest spread.

    Parameters
    ----------
    data : float 2D array of the training data, of shape [N, n_features].
        N must be greater than any of the values in node_indices.
    node_indices : int 1D array of length n_points.  This lists the indices of
        each of the points within the current node.

    Returns
    -------
    i_max : int
        The index of the feature (dimension) within the node that has the
        largest spread.

    Notes
    -----
    In numpy, this operation is equivalent to

    def find_node_split_dim(data, node_indices):
        return np.argmax(data[node_indices].max(0) - data[node_indices].min(0))

    The cython version is much more efficient in both computation and memory.
    """
    n_points = node_indices.size
    n_features = data.shape[1]

    j_max = 0
    max_spread = 0

    for j in range(n_features):
        max_val = data[node_indices[0], j]
        min_val = max_val
        for i in range(1, n_points):
            val = data[node_indices[i], j]
            if val > max_val:
                max_val = val
            elif val < min_val:
                min_val = val
        spread = max_val - min_val
        if spread > max_spread:
            max_spread = spread
            j_max = j
    return j_max


@nb.njit()
def partition_node_indices(data: FloatArray, node_indices: IntArray,
                           split_dim: int, split_index: int):
    """Partition points in the node into two equal-sized groups.

    Upon return, the values in node_indices will be rearranged such that
    (assuming numpy-style indexing):

        data[node_indices[0:split_index], split_dim]
          <= data[node_indices[split_index], split_dim]

    and

        data[node_indices[split_index], split_dim]
          <= data[node_indices[split_index:n_points], split_dim]

    The algorithm is essentially a partial in-place quicksort around a
    set pivot.

    Parameters
    ----------
    TODO

    """
    left = 0
    right = node_indices.size - 1

    while True:
        midindex = left
        for i in range(left, right):
            d1 = data[node_indices[i], split_dim]
            d2 = data[node_indices[right], split_dim]
            if d1 < d2:
                swap(node_indices, i, midindex)
                midindex += 1
        swap(node_indices, midindex, right)
        if midindex == split_index:
            break
        elif midindex < split_index:
            left = midindex + 1
        else:
            right = midindex - 1


@nb.njit()
def permute_tree(data: np.ndarray, idx_array: IntArray, perm: IntArray):
    n = idx_array.size
    # tmp[perm] = np.arange(n)
    tmp = np.empty((n,), dtype=idx_array.dtype)
    for i in range(n):
        tmp[perm[i]] = i
    permuted_perm = tmp[idx_array]
    return data[perm], permuted_perm


# _spec = (
#     ('n_samples', nb.int64),
#     ('n_features', nb.int64),
#     ('leaf_size', nb.int64),
#     ('n_levels', nb.int64),
#     ('n_nodes', nb.int64),
#     ('data', FLOAT_TYPE_T[:, :]),
#     ('idx_array', INT_TYPE_T[:]),
#     ('idx_start', INT_TYPE_T[:]),
#     ('idx_end', INT_TYPE_T[:]),
#     ('is_leaf', nb.boolean[:]),
#     ('radius', FLOAT_TYPE_T[:]),
# )


class QueryResult(NamedTuple):
    dists: FloatArray
    indices: IntArray
    counts: IntArray


class RejectionSampleResult(NamedTuple):
    indices: IntArray
    count: int


class IFPSampleResult(NamedTuple):
    indices: IntArray
    min_dists: FloatArray
    min_dist: float


class RejectionSampleQueryResult(NamedTuple):
    sample_result: RejectionSampleResult
    query_result: QueryResult


class IFPSampleQueryResult(NamedTuple):
    sample_result: IFPSampleResult
    query_result: QueryResult


class TreeData(NamedTuple):
    n_samples: int
    n_features: int
    leaf_size: int
    n_levels: int
    n_nodes: int
    data: FloatArray
    idx_array: IntArray
    idx_start: IntArray
    idx_end: IntArray
    is_leaf: BoolArray


# @nb.njit(inline='always', fastmath=FASTMATH)
# def min_rdist(node_bounds, i_node, x):
#     """Compute the minimum reduced-distance between a point and a node"""
#     rdist = 0.0

#     for j in range(x.size):
#         d_lo = node_bounds[0, i_node, j] - x[j]
#         d_hi = x[j] - node_bounds[1, i_node, j]
#         d = ((d_lo + abs(d_lo)) + (d_hi + abs(d_hi))) / 2
#         rdist += d * d

#     return rdist

# @nb.njit(inline='always', fastmath=FASTMATH)
# def min_dist(node_bounds, i_node, pt):
#     return pow(min_rdist(node_bounds, i_node, pt), 0.5)

# @nb.njit(inline='always', fastmath=FASTMATH)
# def max_rdist(node_bounds, i_node, x):
#     """Compute the maximum reduced-distance between a point and a node"""
#     rdist = 0.0

#     for j in range(x.ize):
#         d_lo = abs(x[j] - node_bounds[0, i_node, j])
#         d_hi = abs(x[j] - node_bounds[1, i_node, j])
#         d = max(d_lo, d_hi)
#         rdist += d * d

#     return rdist

# @nb.njit(inline='always', fastmath=FASTMATH)
# def max_dist(node_bounds, i_node, x):
#     """Compute the maximum distance between a point and a node"""
#     return pow(max_rdist(node_bounds, i_node, x), 0.5)

# @nb.njit(inline='always', fastmath=FASTMATH)
# def _min_max_rdist(node_bounds, i_node, x):
#     """Compute the minimum and maximum distance between a point and a node"""

#     min_dist = 0.0
#     max_dist = 0.0

#     for j in range(x.size):
#         d_lo = node_bounds[0, i_node, j] - x[j]
#         d_hi = x[j] - node_bounds[1, i_node, j]
#         d = (d_lo + abs(d_lo)) + (d_hi + abs(d_hi))
#         min_dist += pow(0.5 * d, 2)
#         max_dist += pow(max(abs(d_lo), abs(d_hi)), 2)
#
# return min_dist, max_dist


@nb.njit(parallel=True, inline='always')
def arange(length, dtype=INT_TYPE):
    out = np.empty((length,), dtype=dtype)
    for i in nb.prange(length):  # pylint: disable=not-an-iterable
        out[i] = i
    return out


@nb.njit(parallel=True)
def get_tree_data(data: FloatArray,
                  leaf_size: int = 40,
                  int_type=INT_TYPE,
                  bool_type=BOOL_TYPE):
    # validate data
    if data.size == 0:
        raise ValueError("X is an empty array")

    if leaf_size < 1:
        raise ValueError("leaf_size must be greater than or equal to 1")

    n_samples, n_features = data.shape

    # CHANGE: (n_samples - 1) -> n_samples
    n_levels = 1 + int(np.log2(max(1, n_samples / leaf_size)))
    n_nodes = np.power(2, n_levels) - 1
    # self.idx_array = np.arange(self.n_samples, dtype=int_type)
    idx_array = arange(n_samples, dtype=int_type)

    idx_start = np.zeros((n_nodes,), dtype=int_type)
    idx_end = np.zeros((n_nodes,), dtype=int_type)
    is_leaf = np.zeros((n_nodes,), dtype=bool_type)
    # radius = np.zeros((n_nodes,), dtype=float_type)

    tree_data = TreeData(
        n_samples=n_samples,
        n_features=n_features,
        leaf_size=leaf_size,
        n_levels=n_levels,
        n_nodes=n_nodes,
        data=data,
        idx_array=idx_array,
        idx_start=idx_start,
        idx_end=idx_end,
        is_leaf=is_leaf,
    )
    _recursive_build(0, 0, n_samples, leaf_size, n_nodes, data, idx_array,
                     idx_start, idx_end, is_leaf)
    return tree_data


@nb.njit()
def _recursive_build(
        i_node: int,
        idx_start_value: int,
        idx_end_value: int,
        leaf_size: int,
        n_nodes: int,
        data: FloatArray,
        idx_array: IntArray,
        idx_start: IntArray,
        idx_end: IntArray,
        is_leaf: BoolArray,
):
    """Recursively build the tree.

    Parameters
    ----------
    tree: TreeData
    i_node : int
        the node for the current step
    idx_start, idx_end : int
        the bounding indices in the idx_array which define the points that
        belong to this node.
    """
    n_points = idx_end_value - idx_start_value
    n_mid = n_points // 2
    idx_array_slice = idx_array[idx_start_value:idx_end_value]
    data = data

    # initialize node data
    # self._init_node(i_node, idx_start, idx_end)
    idx_start[i_node] = idx_start_value
    idx_end[i_node] = idx_end_value

    if 2 * i_node + 1 >= n_nodes:
        is_leaf[i_node] = True
        if n_points > 2 * leaf_size:
            # this shouldn't happen if our memory allocation is correct
            # we'll proactively prevent memory errors, but raise a
            # warning saying we're doing so.
            raise Exception(
                'Internal memory layout is flawed: not enough nodes allocated')
            # import warnings
            # warnings.warn("Internal: memory layout is flawed: "
            #               "not enough nodes allocated")

    elif n_points < 2:
        # again, this shouldn't happen if our memory allocation
        # is correct.  Raise a warning.
        raise Exception(
            'Internal memory layout is flawed: too many nodes allocated')
        # import warnings
        # warnings.warn("Internal: memory layout is flawed: "
        #               "too many nodes allocated")
        # self.is_leaf[i_node] = True

    else:
        # split node and recursively construct child nodes.
        is_leaf[i_node] = False
        i_max = find_node_split_dim(data, idx_array_slice)
        partition_node_indices(data, idx_array_slice, i_max, n_mid)
        idx_mid_value = idx_start_value + n_mid
        _recursive_build(2 * i_node + 1, idx_start_value, idx_mid_value,
                         leaf_size, n_nodes, data, idx_array, idx_start,
                         idx_end, is_leaf)
        _recursive_build(2 * i_node + 2, idx_mid_value, idx_end_value,
                         leaf_size, n_nodes, data, idx_array, idx_start,
                         idx_end, is_leaf)


@nb.njit()
def rejection_ifp_sample_query_prealloc(
        rejection_r: float,
        query_r: float,
        start_nodes: IntArray,
        # ----- pre-allocated data
        sample_indices: IntArray,
        dists: FloatArray,
        query_indices: IntArray,
        counts: IntArray,
        consumed: BoolArray,
        min_dists: FloatArray,
        # ----- tree data
        data: FloatArray,
        idx_array: IntArray,
        idx_start: IntArray,
        idx_end: IntArray,
        is_leaf: BoolArray,
        node_data: NodeData,
        rdist: RDist,
        min_max_rdist: MinMaxRDist,
) -> float:
    # initial rejection sample
    sample_size, max_counts = dists.shape
    count = rejection_sample_query_prealloc(
        rejection_r,
        query_r,
        start_nodes,
        sample_indices,
        dists,
        query_indices,
        counts,
        consumed,
        data,
        idx_array,
        idx_start,
        idx_end,
        is_leaf,
        node_data=node_data,
        rdist=rdist,
        min_max_rdist=min_max_rdist,
    )
    if count == sample_size:
        return np.inf

    # update min_dists
    for i in range(count):
        c = counts[i]
        di = dists[i]
        ii = query_indices[i]
        for j in nb.prange(c):  # pylint: disable=not-an-iterable
            dij = di[j]
            iij = ii[j]
            if dij < min_dists[iij]:
                min_dists[iij] = dij

    # construct heap
    n_samples = data.shape[0]
    min_dists *= -1
    heap = ih.padded_index_heap(min_dists, arange(n_samples),
                                (sample_size - count) * max_counts + n_samples)
    heap.heapify()
    # heap = list(zip(min_dists, arange(n_samples)))
    # heapq.heapify(heap)
    min_dists *= -1

    # ifp sample
    return ifp_sample_query_prealloc(
        query_r,
        start_nodes,
        sample_indices[count:],
        dists,
        query_indices,
        counts[count:],
        consumed,
        min_dists,
        heap,
        n_samples,
        data,
        idx_array,
        idx_start,
        idx_end,
        is_leaf,
        node_data=node_data,
        rdist=rdist,
        min_max_rdist=min_max_rdist,
    )


# issues with parallel==True and heapq or zip?
@nb.njit()
def ifp_sample_query_prealloc(
        query_r: float,
        start_nodes: IntArray,
        # -----
        # pre-allocated data
        sample_indices: IntArray,
        dists: FloatArray,
        query_indices: IntArray,
        counts: IntArray,
        consumed: BoolArray,
        min_dists: FloatArray,  # in_size, minimum distances
        heap: ih.IndexHeap,  # heapified IndexHeap
        # -----
        # tree data
        n_samples: int,
        data: FloatArray,
        idx_array: IntArray,
        idx_start: IntArray,
        idx_end: IntArray,
        is_leaf: BoolArray,
        node_data: NodeData,
        rdist: RDist,
        min_max_rdist: MinMaxRDist,
        eps: float = 1e-4) -> float:
    count = 0
    sample_size = sample_indices.size
    _, max_neighbors = dists.shape
    top_dist = -np.inf
    while heap.length > 0:
        top_dist, index = heap.pop()
        min_dist = min_dists[index]
        if np.isfinite(min_dist):
            diff = abs(min_dist + top_dist)  # top dist is negative
            if diff > eps:
                continue
        sample_indices[count] = index
        di = dists[count]
        ii = query_indices[count]
        # populate di, ii
        instance_count = counts[count] = _query_radius_single_bottom_up(
            0,
            max_neighbors,
            start_nodes[index],
            data[index],
            di,
            ii,
            query_r,
            data,
            idx_array,
            idx_start,
            idx_end,
            is_leaf,
            node_data=node_data,
            rdist=rdist,
            min_max_rdist=min_max_rdist,
        )
        count += 1
        if count >= sample_size:
            break
        for k in range(instance_count):
            dik = di[k]
            iik = ii[k]
            old_dist = min_dists[iik]
            if dik < old_dist:
                min_dists[iik] = dik
                heap.push(-dik, iik)
    else:
        raise RuntimeError('Should have broken...')
    return -top_dist


@nb.njit()
def rejection_sample_precomputed_prealloc(query_indices: IntArray,
                                          counts: IntArray,
                                          sample_indices: IntArray,
                                          consumed: BoolArray) -> int:
    max_size = sample_indices.shape[0]
    if max_size == 0:
        return 0
    sample_count = 0
    in_size = consumed.size
    for i in range(in_size):
        if consumed[i] == 0:
            qi = query_indices[i]
            count = counts[i]
            for j in nb.prange(count):  # pylint: disable=not-an-iterable
                consumed[qi[j]] = 1
            sample_indices[sample_count] = i
            sample_count += 1
            if sample_count >= max_size:
                break
    return sample_count


@nb.njit(inline='always')
def rejection_sample_precomputed(query_indices: IntArray,
                                 counts: IntArray,
                                 max_samples: int,
                                 n_samples: int,
                                 int_type=INT_TYPE,
                                 bool_type=BOOL_TYPE) -> RejectionSampleResult:

    sample_indices = np.full((max_samples,), -1, dtype=int_type)
    consumed = np.zeros((n_samples,), dtype=bool_type)
    count = rejection_sample_precomputed_prealloc(query_indices, counts,
                                                  sample_indices, consumed)
    return RejectionSampleResult(sample_indices, count)


@nb.njit(parallel=False)
def rejection_sample_query_prealloc(
        rejection_r: float,
        query_r: float,
        start_nodes: IntArray,
        sample_indices: IntArray,
        dists: FloatArray,
        query_indices: IntArray,
        counts: IntArray,
        consumed: BoolArray,
        data: FloatArray,
        idx_array: IntArray,
        idx_start: IntArray,
        idx_end: IntArray,
        is_leaf: BoolArray,
        node_data: NodeData,
        rdist: RDist,
        min_max_rdist: MinMaxRDist,
) -> int:
    n_samples = data.shape[0]
    max_samples, max_count = dists.shape
    if max_samples == 0:
        return 0
    sample_count = 0
    for i in range(n_samples):
        if not consumed[i]:
            sample_indices[sample_count] = i
            counts[sample_count] = _rejection_sample_query_single_bottom_up(
                0,
                max_count,
                start_nodes[i],
                data[i],
                dists[sample_count],
                query_indices[sample_count],
                consumed,
                rejection_r,
                query_r,
                data,
                idx_array,
                idx_start,
                idx_end,
                is_leaf,
                node_data,
                rdist=rdist,
                min_max_rdist=min_max_rdist,
            )
            sample_count += 1
            if sample_count >= max_samples:
                break
    return sample_count


@nb.njit(parallel=True)
def get_node_indices(n_samples: int, n_nodes: int, idx_array: IntArray,
                     idx_start: IntArray, idx_end: IntArray,
                     is_leaf: BoolArray):
    nodes = np.empty((n_samples,), dtype=idx_start.dtype)
    for i in nb.prange(n_nodes):  # pylint: disable=not-an-iterable
        if is_leaf[i]:
            nodes[idx_array[idx_start[i]:idx_end[i]]] = i
    return nodes


@nb.njit()
def _rejection_sample_query_single_bottom_up(
        count: int,
        max_count: int,
        i_node: int,
        x: FloatArray,
        dists: FloatArray,
        indices: IntArray,
        consumed: BoolArray,
        rejection_r: float,
        query_r: float,
        # ----- tree data
        data: FloatArray,
        idx_array: IntArray,
        idx_start: IntArray,
        idx_end: IntArray,
        is_leaf: BoolArray,
        node_data: NodeData,
        rdist: RDist,
        min_max_rdist: MinMaxRDist,
):
    count = _query_radius_single_bottom_up(
        count,
        max_count,
        i_node,
        x,
        dists,
        indices,
        query_r,
        data,
        idx_array,
        idx_start,
        idx_end,
        is_leaf,
        node_data=node_data,
        rdist=rdist,
        min_max_rdist=min_max_rdist,
    )
    if rejection_r >= query_r:
        # don't bother doing distance check.
        for i in nb.prange(count):  # pylint: disable=not-an-iterable
            consumed[indices[i]] = True
    else:
        for i in nb.prange(count):  # pylint: disable=not-an-iterable
            if dists[i] < rejection_r:
                consumed[indices[i]] = True
    return count


@nb.njit(parallel=True)
def query_radius_bottom_up_prealloc(
        X: FloatArray,
        r: float,
        start_nodes: IntArray,
        dists: FloatArray,
        indices: IntArray,
        counts: IntArray,
        data: FloatArray,
        idx_array: IntArray,
        idx_start: IntArray,
        idx_end: IntArray,
        is_leaf: BoolArray,
        node_data: NodeData,
        rdist: RDist,
        min_max_rdist: MinMaxRDist,
):
    max_counts = min(dists.shape[1], data.shape[0])
    if max_counts == 0:
        return
    for i in nb.prange(X.shape[0]):  # pylint: disable=not-an-iterable
        counts[i] = _query_radius_single_bottom_up(
            0,
            max_counts,
            start_nodes[i],
            X[i],
            dists[i],
            indices[i],
            r,
            data,
            idx_array,
            idx_start,
            idx_end,
            is_leaf,
            node_data=node_data,
            rdist=rdist,
            min_max_rdist=min_max_rdist,
        )


@nb.njit()
def _query_radius_single_bottom_up(
        count: int,
        max_count: int,
        i_node: int,
        x: FloatArray,
        dists: FloatArray,
        indices: IntArray,
        r: float,
        # -------- tree data
        data: FloatArray,
        idx_array: IntArray,
        idx_start: IntArray,
        idx_end: IntArray,
        is_leaf: BoolArray,
        node_data: NodeData,
        rdist: RDist,
        min_max_rdist: MinMaxRDist,
) -> int:
    count = _query_radius_single(
        count,
        max_count,
        i_node,
        x,
        dists,
        indices,
        r,
        data,
        idx_array,
        idx_start,
        idx_end,
        is_leaf,
        node_data=node_data,
        rdist=rdist,
        min_max_rdist=min_max_rdist,
    )
    while count < max_count and i_node != 0:
        parent = (i_node - 1) // 2
        sibling = i_node + 1 if i_node % 2 else i_node - 1
        count = _query_radius_single(
            count,
            max_count,
            sibling,
            x,
            dists,
            indices,
            r,
            data,
            idx_array,
            idx_start,
            idx_end,
            is_leaf,
            node_data=node_data,
            rdist=rdist,
            min_max_rdist=min_max_rdist,
        )
        i_node = parent
    return count


@nb.njit(parallel=True)
def query_radius_prealloc(
        X: FloatArray,
        r: float,
        dists: FloatArray,
        indices: IntArray,
        counts: IntArray,
        data: FloatArray,
        idx_array: IntArray,
        idx_start: IntArray,
        idx_end: IntArray,
        is_leaf: BoolArray,
        node_data: NodeData,
        rdist: RDist,
        min_max_rdist: MinMaxRDist,
):
    max_results = min(dists.shape[1], data.shape[0])
    if max_results == 0:
        return
    for i in nb.prange(X.shape[0]):  # pylint: disable=not-an-iterable
        counts[i] = _query_radius_single(
            0,
            max_results,
            0,
            X[i],
            dists[i],
            indices[i],
            r,
            data,
            idx_array,
            idx_start,
            idx_end,
            is_leaf,
            node_data=node_data,
            rdist=rdist,
            min_max_rdist=min_max_rdist,
        )


@nb.njit()
def _query_radius_single(
        count: int,
        max_count: int,
        i_node: int,
        x: FloatArray,
        dists: FloatArray,
        indices: IntArray,
        r: float,
        data: FloatArray,
        idx_array: IntArray,
        idx_start: IntArray,
        idx_end: IntArray,
        is_leaf: BoolArray,
        node_data: NodeData,
        rdist: RDist,
        min_max_rdist: MinMaxRDist,
) -> int:
    if count >= max_count:
        return count

    rdist_LB, rdist_UB = min_max_rdist(node_data[i_node], x)

    #------------------------------------------------------------
    # Case 1: all node points are outside distance r.
    #         prune this branch.
    if rdist_LB > r:
        pass

    #------------------------------------------------------------
    # Case 2: all node points are within distance r
    #         add all points to neighbors
    elif rdist_UB <= r:
        for i in range(idx_start[i_node], idx_end[i_node]):
            index = idx_array[i]
            indices[count] = index
            dists[count] = rdist(x, data[index])
            count += 1
            if count >= max_count:
                break

    #------------------------------------------------------------
    # Case 3: this is a leaf node.  Go through all points to
    #         determine if they fall within radius
    elif is_leaf[i_node]:

        for i in range(idx_start[i_node], idx_end[i_node]):
            rdist_x = rdist(x, data[idx_array[i]])
            if rdist_x <= r:
                indices[count] = idx_array[i]
                dists[count] = rdist_x
                count += 1
                if count >= max_count:
                    break

    #------------------------------------------------------------
    # Case 4: Node is not a leaf.  Recursively query subnodes
    else:
        count = _query_radius_single(
            count,
            max_count,
            2 * i_node + 1,
            x,
            dists,
            indices,
            r,
            data,
            idx_array,
            idx_start,
            idx_end,
            is_leaf,
            node_data=node_data,
            rdist=rdist,
            min_max_rdist=min_max_rdist,
        )
        count = _query_radius_single(
            count,
            max_count,
            2 * i_node + 2,
            x,
            dists,
            indices,
            r,
            data,
            idx_array,
            idx_start,
            idx_end,
            is_leaf,
            node_data=node_data,
            rdist=rdist,
            min_max_rdist=min_max_rdist,
        )
    return count


def tree_spec(float_type=FLOAT_TYPE, int_type=INT_TYPE, bool_type=BOOL_TYPE):
    float_type_t = nb.from_dtype(float_type)
    int_type_t = nb.from_dtype(int_type)
    bool_type_t = nb.from_dtype(bool_type)
    return [
        ('n_samples', INT_TYPE_T),
        ('n_features', INT_TYPE_T),
        ('leaf_size', INT_TYPE_T),
        ('n_levels', INT_TYPE_T),
        ('n_nodes', INT_TYPE_T),
        ('data', float_type_t[:, ::1]),
        ('idx_array', int_type_t[::1]),
        ('idx_start', int_type_t[::1]),
        ('idx_end', int_type_t[::1]),
        ('is_leaf', bool_type_t[::1]),
    ]


class BinaryTree(object):

    def __init__(self, data: FloatArray, leaf_size: int = 40):
        self._init(data, leaf_size)

    def _init(self, data: FloatArray, leaf_size: int = 40):
        # assert (data.dtype == self.float_type)
        tree_data = get_tree_data(data,
                                  leaf_size,
                                  int_type=self.int_type,
                                  bool_type=self.bool_type)
        self.n_samples = tree_data.n_samples
        self.n_features = tree_data.n_features
        self.leaf_size = tree_data.leaf_size
        self.n_levels = tree_data.n_levels
        self.n_nodes = tree_data.n_nodes
        self.data = tree_data.data
        self.idx_array = tree_data.idx_array
        self.idx_start = tree_data.idx_start
        self.idx_end = tree_data.idx_end
        self.is_leaf = tree_data.is_leaf
        self.node_data = self._create_node_data()  # pylint: disable=assignment-from-none

    def _create_node_data(self):
        return None
        # raise NotImplementedError('Abstract method')

    @property
    def float_type(self):
        return np.float32

    @property
    def int_type(self):
        return np.int64

    @property
    def bool_type(self):
        return np.uint8

    @property
    def rdist(self) -> RDist:
        raise NotImplementedError('Abstract method')

    @property
    def min_max_rdist(self) -> MinMaxRDist:
        raise NotImplementedError('Abstract method')

    # def rdist(self, x, y):
    #     raise NotImplementedError('Abstract method')

    # def min_max_rdist(self, lower_bounds, upper_bounds, x, n_features):
    #     raise NotImplementedError('Abstract method')

    def query_radius_prealloc(self, X: np.ndarray, r: float, dists: np.ndarray,
                              indices: np.ndarray, counts: np.ndarray) -> None:
        return query_radius_prealloc(
            X,
            r,
            dists,
            indices,
            counts,
            data=self.data,
            idx_array=self.idx_array,
            idx_start=self.idx_start,
            idx_end=self.idx_end,
            is_leaf=self.is_leaf,
            node_data=self.node_data,
            rdist=self.rdist,
            min_max_rdist=self.min_max_rdist,
        )

    def query_radius(self, X: np.ndarray, r: float,
                     max_count: int) -> QueryResult:
        n_queries, n_features = X.shape
        assert (n_features == self.n_features)
        shape = (n_queries, max_count)
        dists = np.full(shape, np.inf, dtype=self.float_type)
        indices = np.full(shape, self.n_samples, dtype=self.int_type)
        counts = np.empty((n_queries,), dtype=self.int_type)
        self.query_radius_prealloc(X, r, dists, indices, counts)
        return QueryResult(dists, indices, counts)

    def query_radius_bottom_up_prealloc(self, X: FloatArray, r: float,
                                        start_nodes: IntArray,
                                        dists: FloatArray, indices: IntArray,
                                        counts: IntArray) -> None:
        query_radius_bottom_up_prealloc(
            X,
            r,
            start_nodes,
            dists,
            indices,
            counts,
            data=self.data,
            idx_array=self.idx_array,
            idx_start=self.idx_start,
            idx_end=self.idx_end,
            is_leaf=self.is_leaf,
            node_data=self.node_data,
            rdist=self.rdist,
            min_max_rdist=self.min_max_rdist,
        )

    def query_radius_bottom_up(self, X: FloatArray, r: float,
                               start_nodes: IntArray, max_count: int):
        n_queries = X.shape[0]
        dists = np.full((n_queries, max_count), np.inf, dtype=self.float_type)
        indices = np.zeros((n_queries, max_count), dtype=self.int_type)
        counts = np.zeros((n_queries,), dtype=self.int_type)
        self.query_radius_bottom_up_prealloc(X, r, start_nodes, dists, indices,
                                             counts)
        return QueryResult(dists, indices, counts)

    def rejection_sample_query_prealloc(self, rejection_r: float,
                                        query_r: float, start_nodes: IntArray,
                                        sample_indices: IntArray,
                                        dists: FloatArray,
                                        query_indices: IntArray,
                                        counts: IntArray, consumed: BoolArray):
        return rejection_sample_query_prealloc(
            rejection_r,
            query_r,
            start_nodes,
            sample_indices,
            dists,
            query_indices,
            counts,
            consumed,
            data=self.data,
            idx_array=self.idx_array,
            idx_start=self.idx_start,
            idx_end=self.idx_end,
            is_leaf=self.is_leaf,
            node_data=self.node_data,
            rdist=self.rdist,
            min_max_rdist=self.min_max_rdist,
        )

    def rejection_sample_query(self, rejection_r, query_r,
                               start_nodes: IntArray, max_samples: int,
                               max_counts: int) -> RejectionSampleQueryResult:
        sample_indices = np.full((max_samples,),
                                 self.n_samples,
                                 dtype=self.int_type)
        shape = (max_samples, max_counts)
        dists = np.full(shape, np.inf, dtype=self.float_type)
        query_indices = np.full(shape, self.n_samples, dtype=self.int_type)
        counts = np.full((max_samples,), -1, dtype=self.int_type)
        consumed = np.zeros((self.n_samples,), dtype=self.bool_type)
        sample_count = self.rejection_sample_query_prealloc(
            rejection_r, query_r, start_nodes, sample_indices, dists,
            query_indices, counts, consumed)

        return RejectionSampleQueryResult(
            RejectionSampleResult(sample_indices, sample_count),
            QueryResult(dists, query_indices, counts))

    def ifp_sample_query_prealloc(
            self,
            query_r: float,
            start_nodes: IntArray,
            # -----
            # pre-allocated data
            sample_indices: IntArray,
            dists: FloatArray,
            query_indices: IntArray,
            counts: IntArray,
            consumed: BoolArray,
            min_dists: FloatArray,  # in_size, minimum distances
            heap,  # heap, heap-sorted list of (neg_dist, index) tuples
    ) -> float:
        return ifp_sample_query_prealloc(
            query_r,
            start_nodes,
            sample_indices,
            dists,
            query_indices,
            counts,
            consumed,
            min_dists,
            heap,
            self.n_samples,
            self.data,
            self.idx_array,
            self.idx_start,
            self.idx_end,
            self.is_leaf,
            node_data=self.node_data,
            rdist=self.rdist,
            min_max_rdist=self.min_max_rdist,
        )

    def ifp_sample_query(self, query_r: float, start_nodes: IntArray,
                         sample_size: int,
                         max_counts: int) -> IFPSampleQueryResult:
        sample_indices = np.full((sample_size,),
                                 self.n_samples,
                                 dtype=self.int_type)
        shape = (sample_size, max_counts)
        dists = np.full(shape, np.inf, dtype=self.float_type)
        query_indices = np.full(shape, self.n_samples, dtype=self.int_type)
        counts = np.full((sample_size,), -1, dtype=self.int_type)
        consumed = np.zeros((self.n_samples,), dtype=self.bool_type)
        min_dists = np.full((self.n_samples,), -np.inf, dtype=self.float_type)

        # heap = list(zip(min_dists, arange(self.n_samples,)))
        heap = ih.padded_index_heap(
            min_dists,
            arange(self.n_samples, dtype=self.int_type),
            sample_size * max_counts + self.n_samples,
        )
        min_dists *= -1
        min_dist = self.ifp_sample_query_prealloc(query_r, start_nodes,
                                                  sample_indices, dists,
                                                  query_indices, counts,
                                                  consumed, min_dists, heap)

        return IFPSampleQueryResult(
            IFPSampleResult(sample_indices, min_dists, min_dist),
            QueryResult(dists, query_indices, counts))

    def rejection_ifp_sample_query_prealloc(
            self,
            rejection_r: float,
            query_r: float,
            start_nodes: IntArray,
            # -----
            # pre-allocated data
            sample_indices: IntArray,
            dists: FloatArray,
            query_indices: IntArray,
            counts: IntArray,
            consumed: BoolArray,
            min_dists: FloatArray) -> float:
        return rejection_ifp_sample_query_prealloc(
            rejection_r,
            query_r,
            start_nodes,
            sample_indices,
            dists,
            query_indices,
            counts,
            consumed,
            min_dists,
            self.data,
            self.idx_array,
            self.idx_start,
            self.idx_end,
            self.is_leaf,
            node_data=self.node_data,
            rdist=self.rdist,
            min_max_rdist=self.min_max_rdist,
        )

    def rejection_ifp_sample_query(self, rejection_r: float, query_r: float,
                                   start_nodes: IntArray, sample_size: int,
                                   max_counts: int) -> IFPSampleQueryResult:
        sample_indices = np.full((sample_size,),
                                 self.n_samples,
                                 dtype=self.int_type)
        shape = (sample_size, max_counts)
        dists = np.full(shape, np.inf, dtype=self.float_type)
        query_indices = np.full(shape, self.n_samples, dtype=self.int_type)
        counts = np.full((sample_size,), -1, dtype=self.int_type)
        consumed = np.zeros((self.n_samples,), dtype=self.bool_type)
        min_dists = np.full((self.n_samples,), np.inf, dtype=self.float_type)

        min_dist = self.rejection_ifp_sample_query_prealloc(
            rejection_r, query_r, start_nodes, sample_indices, dists,
            query_indices, counts, consumed, min_dists)

        return IFPSampleQueryResult(
            IFPSampleResult(sample_indices, min_dists, min_dist),
            QueryResult(dists, query_indices, counts))

    def get_node_indices(self):
        return get_node_indices(n_samples=self.n_samples,
                                n_nodes=self.n_nodes,
                                idx_array=self.idx_array,
                                idx_start=self.idx_start,
                                idx_end=self.idx_end,
                                is_leaf=self.is_leaf)

    def permute(self, perm):
        self.data, self.idx_array = permute_tree(self.data, self.idx_array,
                                                 perm)
