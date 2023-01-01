import os
from typing import Callable, NamedTuple, Optional, Tuple

import numba as nb
import numpy as np

from numba_neighbors import index_heap as ih

FASTMATH = True
PARALLEL = os.environ.get("NUMBA_PARALLEL", "1") != "0"

INT_TYPE = np.int64
INT_TYPE_T = nb.int64

FLOAT_TYPE = np.float32
FLOAT_TYPE_T = nb.float32

BOOL_TYPE = np.uint8
BOOL_TYPE_T = nb.uint8

IntArray = np.ndarray
FloatArray = np.ndarray
BoolArray = np.ndarray

NodeDataArray = np.ndarray

RDist = Callable[[FloatArray, FloatArray], float]
MinMaxRDist = Callable[[NodeDataArray, FloatArray], Tuple[float, float]]


@nb.njit(inline="always")
def swap(arr, i1, i2):
    """Swap values at index i1 and i2 of arr."""
    tmp = arr[i1]
    arr[i1] = arr[i2]
    arr[i2] = tmp


@nb.njit(inline="always")
def dual_swap(darr, iarr, i1, i2):
    """swap the values at inex i1 and i2 of both darr and iarr"""
    dtmp = darr[i1]
    darr[i1] = darr[i2]
    darr[i2] = dtmp

    itmp = iarr[i1]
    iarr[i1] = iarr[i2]
    iarr[i2] = itmp


@nb.njit()
def _simultaneous_sort(  # pylint:disable=too-many-branches
    priorities: np.ndarray, values: np.ndarray
) -> None:
    """
    Recursively sort the arrays according to priorities in place.

    The same permutation is applied to both `priorities` and `values`. The
    equivalent in numpy (though quite a bit slower) is

    ```python
    def simultaneous_sort(priorities, values):
        i = np.argsort(priorities)
        return priorities[i], values[i]
    ```

    Args:
        priorities: 1D array to sort by
        values: array of values to sort in the same way as priorities.
    """
    # in the small-array case, do things efficiently
    size = priorities.size
    if size <= 1:
        pass
    elif size == 2:
        if priorities[0] > priorities[1]:
            dual_swap(priorities, values, 0, 1)
    elif size == 3:
        if priorities[0] > priorities[1]:
            dual_swap(priorities, values, 0, 1)
        if priorities[1] > priorities[2]:
            dual_swap(priorities, values, 1, 2)
            if priorities[0] > priorities[1]:
                dual_swap(priorities, values, 0, 1)
    else:
        # Determine the pivot using the median-of-three rule.
        # The smallest of the three is moved to the beginning of the array,
        # the middle (the pivot value) is moved to the end, and the largest
        # is moved to the pivot index.
        pivot_idx = size // 2
        if priorities[0] > priorities[size - 1]:
            dual_swap(priorities, values, 0, size - 1)
        if priorities[size - 1] > priorities[pivot_idx]:
            dual_swap(priorities, values, size - 1, pivot_idx)
            if priorities[0] > priorities[size - 1]:
                dual_swap(priorities, values, 0, size - 1)
        pivot_val = priorities[size - 1]

        # partition indices about pivot.  At the end of this operation,
        # pivot_idx will contain the pivot value, everything to the left
        # will be smaller, and everything to the right will be larger.
        store_values = 0
        for i in range(size - 1):
            if priorities[i] < pivot_val:
                dual_swap(priorities, values, i, store_values)
                store_values += 1
        dual_swap(priorities, values, store_values, size - 1)
        pivot_idx = store_values

        # recursively sort each side of the pivot
        if pivot_idx > 1:
            _simultaneous_sort(priorities[:pivot_idx], values[:pivot_idx])

        if pivot_idx + 2 < size:
            start = pivot_idx + 1
            _simultaneous_sort(priorities[start:], values[start:])


@nb.njit(parallel=PARALLEL)
def simultaneous_sort(priorities: np.ndarray, values: np.ndarray) -> None:
    """
    Independently sort the rows of the arrays according to pririties in place.

    The permutation is calculated based on sorting priorities, and the same
    permutation is applied to values per row.

    Args:
        priorities: 2D array
        values: ND array, N >= 2, where priorities.shape == values.shape[:2].
    """
    assert priorities.shape == values.shape[:2]
    assert len(priorities.shape) == 2
    for row in nb.prange(priorities.shape[0]):  # pylint: disable=not-an-iterable
        _simultaneous_sort(priorities[row], values[row])


@nb.njit(parallel=PARALLEL)
def simultaneous_sort_partial(
    priorities: np.ndarray, values: np.ndarray, counts: IntArray
):
    """In-place simultaneous sort the given row of the arrays

    This python wrapper exists primarily to enable unit testing
    of the _simultaneous_sort C routine.
    """
    assert priorities.shape == values.shape
    assert len(priorities.shape) == 2
    assert priorities.shape[:1] == counts.shape
    for row in nb.prange(priorities.shape[0]):  # pylint: disable=not-an-iterable
        count = counts[row]
        _simultaneous_sort(priorities[row, :count], values[row, :count])


@nb.njit()
def find_node_split_dim(data: FloatArray, node_indices: IntArray) -> int:
    """Find the dimension with the largest spread.

    In numpy, this operation is equivalent to

    ```python
    np.argmax(data[node_indices].max(0) - data[node_indices].min(0))
    ```

    or

    ```python
    np.argmax(data[node_indices].ptp())
    ```

    Args:
        data: float 2D array of the training data, of shape [N, n_features].
            N must be greater than any of the values in node_indices.
        node_indices: int 1D array of length n_points.  This lists the indices of
            each of the points within the current node.

    Returns:
        i_max: int, the index of the feature (dimension) within the node that
            has the largest spread.
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
def partition_node_indices(
    data: FloatArray, node_indices: IntArray, split_dim: int, split_index: int
):
    """
    Partition points in the node into two equal-sized groups.

    Upon return, the values in node_indices will be rearranged such that
    (assuming numpy-style indexing):

        data[node_indices[:split_index], split_dim]
          <= data[node_indices[split_index], split_dim]
          <= data[node_indices[split_index:n_points], split_dim]

    The algorithm is essentially a partial in-place quicksort around a
    set pivot.

    Args:
        data: 2D float, [N, n_features] coordinates of points
        node_indices: indices into data that satisfy the above upon returning.
        split_dim: int, dimension to split on.
        split_index: pivot index.
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
        if midindex < split_index:
            left = midindex + 1
        else:
            right = midindex - 1


@nb.njit()
def permute_tree(data: np.ndarray, idx_array: IntArray, perm: IntArray):
    """
    Get data for a permuted tree.

    All BinaryTree operations use data[idx_array]. This operation permutes
    data by perm but also permutes idx_array such that the returned
    (data, idx_array) leaves data[idx_array] unchanged, i.e.

    ```python
    out_data, out_idx_array = permute_data(data, idx_array, perm)
    np.testing.assert_equal(out_data, data[perm])
    np.testing.assert_equal(out_data[out_idx_array], data[idx_array])
    ```

    Args:
        data: 2D float array.
        idx_array: inded array constructed presumably by a binary tree.
        perm: arbitrary 1D int permuatation vector.

    Returns:
        out_data, out_idx_array: permuted data and idx_array such that the
            above conditions are met.
    """
    n = idx_array.size
    # tmp[perm] = np.arange(n)
    tmp = np.empty((n,), dtype=idx_array.dtype)
    for i in range(n):
        tmp[perm[i]] = i
    permuted_perm = tmp[idx_array]
    return data[perm], permuted_perm


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


@nb.njit(parallel=PARALLEL, inline="always")
def arange(length, dtype=INT_TYPE):
    """Simple `np.arange` implementation without start/step."""
    out = np.empty((length,), dtype=dtype)
    for i in nb.prange(length):  # pylint: disable=not-an-iterable
        out[i] = i
    return out


@nb.jit()
def create_tree_data(
    data: FloatArray, leaf_size: int = 40, int_type=INT_TYPE, bool_type=BOOL_TYPE
):
    if data.size == 0:
        raise ValueError("X is an empty array")

    if leaf_size < 1:
        raise ValueError("leaf_size must be greater than or equal to 1")

    n_data = data.shape[0]

    # CHANGE: (n_data - 1) -> n_data
    n_levels = 1 + int(np.log2(max(1, n_data / leaf_size)))
    n_nodes = np.power(2, n_levels) - 1
    # self.idx_array = np.arange(self.n_data, dtype=int_type)
    idx_array = arange(n_data, dtype=int_type)

    idx_start = np.zeros((n_nodes,), dtype=int_type)
    idx_end = np.zeros((n_nodes,), dtype=int_type)
    is_leaf = np.zeros((n_nodes,), dtype=bool_type)
    # radius = np.zeros((n_nodes,), dtype=float_type)

    return idx_array, idx_start, idx_end, is_leaf


@nb.njit()
def fill_tree_data(
    data: FloatArray,
    leaf_size: int,
    idx_array: IntArray,
    idx_start: IntArray,
    idx_end: IntArray,
    is_leaf: BoolArray,
) -> None:
    """
    Get data associated with BinaryTree.

    Args:
        data: [N, n_features] 2D float array of tree points.
        leaf_size: int, number of points in each leaf.
        int_type: dtype of integer arrays used.
        bool_type: dtype of bool arrays used.

    Returns:
        n_levels: int, number of levels of the tree
        n_nodes: number of nodes in the tree
        idx_array: [N] int_type array of integers. data[idx_array] are data points
            in node ordering.
        idx_start: [n_nodes] int_type array of start indices of each node range.
        idx_end: [n_nodes] int_type array of end indices of each node range.
        is_leaf: [n_nodes] bool array indicating which nodes are leaves.
    """
    # validate data
    n_data = data.shape[0]
    n_nodes = idx_start.size
    _recursive_build(
        0, 0, n_data, leaf_size, n_nodes, data, idx_array, idx_start, idx_end, is_leaf
    )


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
    """Recursively build the tree."""
    n_points = idx_end_value - idx_start_value
    n_mid = n_points // 2
    idx_array_slice = idx_array[idx_start_value:idx_end_value]

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
                "Internal memory layout is flawed: not enough nodes allocated"
            )
            # import warnings
            # warnings.warn("Internal: memory layout is flawed: "
            #               "not enough nodes allocated")

    elif n_points < 2:
        # again, this shouldn't happen if our memory allocation
        # is correct.  Raise a warning.
        raise Exception("Internal memory layout is flawed: too many nodes allocated")
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
        _recursive_build(
            2 * i_node + 1,
            idx_start_value,
            idx_mid_value,
            leaf_size,
            n_nodes,
            data,
            idx_array,
            idx_start,
            idx_end,
            is_leaf,
        )
        _recursive_build(
            2 * i_node + 2,
            idx_mid_value,
            idx_end_value,
            leaf_size,
            n_nodes,
            data,
            idx_array,
            idx_start,
            idx_end,
            is_leaf,
        )


@nb.njit(inline="always")
def _update_min_dists(dists, query_indices, counts, count, min_dists):
    for i in range(count):
        c = counts[i]
        di = dists[i]
        ii = query_indices[i]
        for j in nb.prange(c):  # pylint: disable=not-an-iterable
            dij = di[j]
            iij = ii[j]
            if dij < min_dists[iij]:
                min_dists[iij] = dij


@nb.njit()
def rejection_ifp_sample_query(
    rejection_r: float,
    query_r: float,
    start_nodes: IntArray,
    sample_size: int,
    max_counts: int,
    # ----- tree data
    data: FloatArray,
    idx_array: IntArray,
    idx_start: IntArray,
    idx_end: IntArray,
    is_leaf: BoolArray,
    node_data: NodeDataArray,
    rdist: RDist,
    min_max_rdist: MinMaxRDist,
) -> float:
    """
    Rejection-iterative farthest point sampling and querying.

    Results are saved in preallocated arrays.

    Args:
        rejection_r: reduced radius used in initial rejection sample.
        query_r: reduce query radius used for subsequent IFP sampling and
            returned neighbors.
        start_node: int array of node indices for which data coordinates belong.
        *tree_data: data from the input BinaryTree.

    Returns:
        minimum distance of final sampled point. All non-sampled points should
        be within this distane of a sampled point.
    """
    int_type = idx_array.dtype
    bool_type = is_leaf.dtype
    float_type = data.dtype
    n_data = data.shape[0]
    sample_indices = np.full((sample_size,), n_data, dtype=int_type)
    shape = (sample_size, max_counts)
    dists = np.full(shape, np.inf, dtype=float_type)
    query_indices = np.full(shape, n_data, dtype=int_type)
    counts = np.full((sample_size,), -1, dtype=int_type)
    consumed = np.zeros((n_data,), dtype=bool_type)
    min_dists = np.full((n_data,), np.inf, dtype=float_type)

    max_heap_length = sample_size * max_counts + n_data
    heap_priorities = np.empty((max_heap_length,), dtype=float_type)
    heap_indices = np.empty((max_heap_length,), dtype=int_type)

    min_dist = rejection_ifp_sample_query_prealloc(
        rejection_r=rejection_r,
        query_r=query_r,
        start_nodes=start_nodes,
        sample_indices=sample_indices,
        dists=dists,
        query_indices=query_indices,
        counts=counts,
        consumed=consumed,
        min_dists=min_dists,
        heap_priorities=heap_priorities,
        heap_indices=heap_indices,
        data=data,
        idx_array=idx_array,
        idx_start=idx_start,
        idx_end=idx_end,
        is_leaf=is_leaf,
        node_data=node_data,
        rdist=rdist,
        min_max_rdist=min_max_rdist,
    )

    return IFPSampleQueryResult(
        IFPSampleResult(sample_indices, min_dists, min_dist),
        QueryResult(dists, query_indices, counts),
    )


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
    heap_priorities: FloatArray,
    heap_indices: IntArray,
    # ----- tree data
    data: FloatArray,
    idx_array: IntArray,
    idx_start: IntArray,
    idx_end: IntArray,
    is_leaf: BoolArray,
    node_data: NodeDataArray,
    rdist: RDist,
    min_max_rdist: MinMaxRDist,
) -> float:
    """
    Rejection-iterative farthest point sampling and querying.

    Results are saved in preallocated arrays.

    Args:
        rejection_r: reduced radius used in initial rejection sample.
        query_r: reduce query radius used for subsequent IFP sampling and
            returned neighbors.
        start_node: int array of node indices for which data coordinates belong.

        --- preallocated data below
        sample_indices: [sample_size] preallocated int array in which sample
            indices are saved.
        dists: [sample_size, max_neighbors] array in which distances are saved
        query_indices: [sample_size, max_neighbors] array of indices in
            resulting query.
        counts: [sample_size] array of counts of neighbors
        consumed: [in_size] bool array used in initial rejection sample.
        min_dists: [in_size] float array of minimum distances.

        *tree_data: data from the input BinaryTree.

    Returns:
        minimum distance of final sampled point. All non-sampled points should
        be within this distane of a sampled point.
    """
    # initial rejection sample
    sample_size = counts.size
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
    _update_min_dists(dists, query_indices, counts, count, min_dists)

    # construct heap
    n_data = data.shape[0]

    # heap = ih.padded_index_heap(min_dists, arange(n_data),
    #                             (sample_size - count) * max_counts + n_data)
    for i in nb.prange(n_data):  # pylint: disable=not-an-iterable
        heap_priorities[i] = -min_dists[i]
        heap_indices[i] = i
    heap = ih.IndexHeap(heap_priorities, heap_indices, n_data)
    heap.heapify()

    # ifp sample
    return ifp_sample_query_prealloc(
        query_r,
        start_nodes,
        sample_indices[count:],
        dists[count:],
        query_indices[count:],
        counts[count:],
        min_dists,
        heap,
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
def ifp_sample_precomputed_prealloc(
    dists: FloatArray,
    query_indices: IntArray,
    counts: IntArray,
    # --- precomputed data
    sample_indices: IntArray,
    min_dists: FloatArray,
    heap,
    eps: float = 1e-8,
):
    count = 0
    sample_size = sample_indices.size
    top_dist = -np.inf
    while heap.length > 0:
        top_dist, index = heap.pop()
        min_dist = min_dists[index]
        if np.isfinite(min_dist):
            diff = abs(min_dist + top_dist)  # top dist is negative
            if diff > eps:
                continue
        sample_indices[count] = index
        di = dists[index]
        ii = query_indices[index]
        # populate di, ii
        instance_count = counts[index]

        for k in range(instance_count):
            dik = di[k]
            iik = ii[k]
            old_dist = min_dists[iik]
            if dik < old_dist:
                min_dists[iik] = dik
                heap.push(-dik, iik)
        count += 1
        if count >= sample_size:
            break
    else:
        raise RuntimeError("Should have broken...")
    return -top_dist


@nb.njit(fastmath=True)
def ifp_sample_precomputed(
    dists: FloatArray,
    query_indices: IntArray,
    counts: IntArray,
    sample_size: int,
    eps=1e-8,
) -> IFPSampleResult:
    in_size, max_counts = dists.shape
    int_type = query_indices.dtype
    sample_indices = np.empty((sample_size,), dtype=int_type)
    min_dists = np.full((in_size,), -np.inf, dtype=np.float32)
    heap = ih.padded_index_heap(
        min_dists, arange(in_size, dtype=int_type), sample_size * max_counts + in_size
    )
    min_dists *= -1

    min_dist = ifp_sample_precomputed_prealloc(
        dists, query_indices, counts, sample_indices, min_dists, heap, eps=eps
    )
    return IFPSampleResult(sample_indices, min_dists, min_dist)


@nb.njit(fastmath=True)
def rejection_ifp_sample_precomputed_prealloc(
    dists: FloatArray,
    query_indices: IntArray,
    counts: IntArray,
    # -- prealloc
    sample_indices: IntArray,
    min_dists: FloatArray,
    consumed: BoolArray,
    eps: float = 1e-8,
) -> float:
    in_size, max_counts = dists.shape
    count = rejection_sample_precomputed_prealloc(
        query_indices, counts, sample_indices, consumed
    )
    si = sample_indices[:count]
    _update_min_dists(dists[si], query_indices[si], counts[si], count, min_dists)
    if count == sample_indices.size:
        return np.inf
    min_dists *= -1
    heap = ih.padded_index_heap(
        min_dists,
        arange(in_size, dtype=sample_indices.dtype),
        sample_indices.size * max_counts + in_size,
    )
    heap.heapify()
    min_dists *= -1
    min_dist = ifp_sample_precomputed_prealloc(
        dists, query_indices, counts, sample_indices[count:], min_dists, heap, eps
    )
    return min_dist


@nb.njit(fastmath=True)
def rejection_ifp_sample_precomputed(
    dists: FloatArray,
    query_indices: IntArray,
    counts: IntArray,
    sample_size: int,
    bool_type=BOOL_TYPE,
    eps=1e-8,
) -> IFPSampleResult:
    in_size = counts.size

    sample_indices = np.empty((sample_size,), dtype=query_indices.dtype)
    min_dists = np.full((in_size,), np.inf, dtype=dists.dtype)
    consumed = np.zeros((in_size,), dtype=bool_type)
    min_dist = rejection_ifp_sample_precomputed_prealloc(
        dists, query_indices, counts, sample_indices, min_dists, consumed, eps
    )
    return IFPSampleResult(sample_indices, min_dists, min_dist)


@nb.njit()
def ifp_sample_query(
    query_r: float,
    start_nodes: IntArray,
    sample_size: int,
    max_counts: int,
    # tree data
    data: FloatArray,
    idx_array: IntArray,
    idx_start: IntArray,
    idx_end: IntArray,
    is_leaf: BoolArray,
    node_data: NodeDataArray,
    rdist: RDist,
    min_max_rdist: MinMaxRDist,
    eps: float = 1e-8,
) -> IFPSampleQueryResult:
    n_data = data.shape[0]
    int_type = idx_array.dtype
    float_type = data.dtype
    sample_indices = np.full((sample_size,), n_data, dtype=int_type)
    shape = (sample_size, max_counts)
    dists = np.full(shape, np.inf, dtype=float_type)
    query_indices = np.full(shape, n_data, dtype=int_type)
    counts = np.full((sample_size,), -1, dtype=int_type)
    min_dists = np.full((n_data,), -np.inf, dtype=float_type)

    # heap = list(zip(min_dists, arange(self.n_data,)))
    heap = ih.padded_index_heap(
        min_dists, arange(n_data, dtype=int_type), sample_size * max_counts + n_data,
    )
    min_dists *= -1
    min_dist = ifp_sample_query_prealloc(
        query_r=query_r,
        start_nodes=start_nodes,
        sample_indices=sample_indices,
        dists=dists,
        query_indices=query_indices,
        counts=counts,
        min_dists=min_dists,
        heap=heap,
        data=data,
        idx_array=idx_array,
        idx_start=idx_start,
        idx_end=idx_end,
        is_leaf=is_leaf,
        node_data=node_data,
        rdist=rdist,
        min_max_rdist=min_max_rdist,
        eps=eps,
    )

    return IFPSampleQueryResult(
        IFPSampleResult(sample_indices, min_dists, min_dist),
        QueryResult(dists, query_indices, counts),
    )


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
    min_dists: FloatArray,  # in_size, minimum distances
    heap: ih.IndexHeap,  # heapified IndexHeap
    # -----
    # tree data
    data: FloatArray,
    idx_array: IntArray,
    idx_start: IntArray,
    idx_end: IntArray,
    is_leaf: BoolArray,
    node_data: NodeDataArray,
    rdist: RDist,
    min_max_rdist: MinMaxRDist,
    eps: float = 1e-8,
) -> float:
    """
    Perform iterative farthest point sampling and querying.

    Results are saved into preallocated arrays.

    Args:
        query_r: float, reduced query radius.
        start_nodes: int array, node indices of tree data.
        sample_indices, dists, query_indices, counts, in_dists, heap:
            preallocated data
        *tree_data: data from the input BinaryTree
        eps: float, the amount by which min_dist must be different to saved
            distance in priority queue.

    Returns:
        minimum distance of final sampled point. All points should be within
        this distance of a sampled point.
    """
    count = 0
    sample_size = sample_indices.size
    _, max_neighbors = dists.shape
    top_dist = -np.inf
    while heap.length > 0:
        top_dist, index = heap.pop()
        min_dist = min_dists[index]
        if np.isfinite(min_dist):
            diff = abs(min_dist + top_dist)  # top_dist is negative
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
        for k in range(instance_count):
            dik = di[k]
            iik = ii[k]
            old_dist = min_dists[iik]
            if dik < old_dist:
                min_dists[iik] = dik
                heap.push(-dik, iik)
        count += 1
        if count >= sample_size:
            break
    else:
        raise RuntimeError("Should have broken...")
    return -top_dist


@nb.njit()
def rejection_sample_precomputed_prealloc(
    query_indices: IntArray,
    counts: IntArray,
    sample_indices: IntArray,
    consumed: BoolArray,
    valid: Optional[BoolArray] = None,
) -> int:
    """
    Perform rejection  sampling with precomputed sample indices.

    Args:
        query_indices: [in_size, max_neighbors] neighbors of each input point.
        counts: [in_size] number of valid indices for each row of query_indices.
        sample_indices: [max_sample_size] preallocated int array.
        consumed: [in_size] preallocated bool array.
        valid: [in_size, max_neighbors] optional bool array. If given, any
            false value will result in the corresponding query_indices being
            ignored.

    Returns:
        count: number of points sampled.
    """
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
                if valid is not None and not valid[i, j]:
                    continue
                consumed[qi[j]] = 1
            sample_indices[sample_count] = i
            sample_count += 1
            if sample_count >= max_size:
                break
    return sample_count


@nb.njit(inline="always")
def rejection_sample_precomputed(
    query_indices: IntArray,
    counts: IntArray,
    max_samples: Optional[int],
    int_type=INT_TYPE,
    bool_type=BOOL_TYPE,
    valid: Optional[BoolArray] = None,
) -> RejectionSampleResult:
    """
    Perform rejection sampling with precomputed sample indices.

    Args:
        query_indices: [in_size, max_neighbors] neighbors of each input point.
        counts: [in_size] number of valid indices for each row of query_indices.
        max_samples: int, maximum number of samples to consider.
        int_type: int dtype
    """
    if max_samples is None:
        max_samples = counts.shape[0]
    in_size = counts.size
    sample_indices = np.full((max_samples,), -1, dtype=int_type)
    consumed = np.zeros((in_size,), dtype=bool_type)
    count = rejection_sample_precomputed_prealloc(
        query_indices, counts, sample_indices, consumed, valid
    )
    return RejectionSampleResult(sample_indices, count)


@nb.njit()
def rejection_sample_query(
    rejection_r,
    query_r,
    start_nodes: IntArray,
    max_samples: int,
    max_counts: int,
    # --- tree data arrays below
    data: FloatArray,
    idx_array: IntArray,
    idx_start: IntArray,
    idx_end: IntArray,
    is_leaf: BoolArray,
    node_data: NodeDataArray,
    rdist: RDist,
    min_max_rdist: MinMaxRDist,
) -> IFPSampleQueryResult:
    n_data = data.shape[0]
    int_type = idx_array.dtype
    float_type = data.dtype
    bool_type = is_leaf.dtype
    sample_indices = np.full((max_samples,), n_data, dtype=int_type)
    shape = (max_samples, max_counts)
    dists = np.full(shape, np.inf, dtype=float_type)
    query_indices = np.full(shape, n_data, dtype=int_type)
    counts = np.full((max_samples,), -1, dtype=int_type)
    consumed = np.zeros((n_data,), dtype=bool_type)
    sample_count = rejection_sample_query_prealloc(
        rejection_r,
        query_r,
        start_nodes,
        sample_indices,
        dists,
        query_indices,
        counts,
        consumed,
        data=data,
        idx_array=idx_array,
        idx_start=idx_start,
        idx_end=idx_end,
        is_leaf=is_leaf,
        node_data=node_data,
        rdist=rdist,
        min_max_rdist=min_max_rdist,
    )

    return RejectionSampleQueryResult(
        RejectionSampleResult(sample_indices, sample_count),
        QueryResult(dists, query_indices, counts),
    )


@nb.njit()
def rejection_sample_query_prealloc(
    rejection_r: float,
    query_r: float,
    start_nodes: IntArray,
    # --- preallocated arrays below
    sample_indices: IntArray,
    dists: FloatArray,
    query_indices: IntArray,
    counts: IntArray,
    consumed: BoolArray,
    # --- tree data arrays below
    data: FloatArray,
    idx_array: IntArray,
    idx_start: IntArray,
    idx_end: IntArray,
    is_leaf: BoolArray,
    node_data: NodeDataArray,
    rdist: RDist,
    min_max_rdist: MinMaxRDist,
) -> int:
    """
    Perform simultaneous rejection sampling and querying.

    Data saved to preallocated arrays.

    Args:
        rejection_r: reduced radius used in rejection sampling.
        query_r: reduced radius used in query.
        start_nodes: starting index nodes of each point.
        *preallocated arrays
        *tree_data

    Returns:
        number of points sampled.
    """
    n_data = data.shape[0]
    max_samples, max_count = dists.shape
    if max_samples == 0:
        return 0
    sample_count = 0
    for i in range(n_data):
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


@nb.njit()
def get_node_indices_prealloc(
    idx_array: IntArray,
    idx_start: IntArray,
    idx_end: IntArray,
    is_leaf: BoolArray,
    node_indices: IntArray,
) -> None:
    n_nodes = is_leaf.size
    for i in range(n_nodes):  # pylint: disable=not-an-iterable
        if is_leaf[i]:
            node_indices[idx_array[idx_start[i] : idx_end[i]]] = i


@nb.njit()
def get_node_indices(
    idx_array: IntArray, idx_start: IntArray, idx_end: IntArray, is_leaf: BoolArray
) -> IntArray:
    """Get the index of the leaf of each data point."""
    node_indices = np.empty((idx_array.size,), dtype=idx_start.dtype)
    get_node_indices_prealloc(idx_array, idx_start, idx_end, is_leaf, node_indices)
    return node_indices


@nb.njit(parallel=PARALLEL)
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
    node_data: NodeDataArray,
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


@nb.njit(parallel=PARALLEL)
def query_radius_bottom_up(
    X: FloatArray,
    r: float,
    start_nodes: IntArray,
    max_count: int,
    # --- tree data below
    data: FloatArray,
    idx_array: IntArray,
    idx_start: IntArray,
    idx_end: IntArray,
    is_leaf: BoolArray,
    node_data: NodeDataArray,
    rdist: RDist,
    min_max_rdist: MinMaxRDist,
):
    n_queries = X.shape[0]
    dists = np.full((n_queries, max_count), np.inf, dtype=data.dtype)
    indices = np.full((n_queries, max_count), data.shape[0], dtype=idx_array.dtype)
    counts = np.zeros((n_queries,), dtype=idx_start.dtype)
    query_radius_bottom_up_prealloc(
        X,
        r,
        start_nodes,
        dists,
        indices,
        counts,
        data=data,
        idx_array=idx_array,
        idx_start=idx_start,
        idx_end=idx_end,
        is_leaf=is_leaf,
        node_data=node_data,
        rdist=rdist,
        min_max_rdist=min_max_rdist,
    )
    return QueryResult(dists, indices, counts)


@nb.njit(parallel=PARALLEL)
def query_radius_bottom_up_prealloc(
    X: FloatArray,
    r: float,
    start_nodes: IntArray,
    # --- preallocated data below
    dists: FloatArray,
    indices: IntArray,
    counts: IntArray,
    # --- tree data below
    data: FloatArray,
    idx_array: IntArray,
    idx_start: IntArray,
    idx_end: IntArray,
    is_leaf: BoolArray,
    node_data: NodeDataArray,
    rdist: RDist,
    min_max_rdist: MinMaxRDist,
):
    """
    Query a binary tree when the leaf index of each query point is known.

    Args:
        X: [n_queries, n_features] query points
        r: float
        start_nodes: [n_queries] node index of the containing leaf of each
            point in X. Results should still be accurate if these are incorrect,
            though computation time may be greater.
        dists, indices, counts: preallocated data
        *tree_data
    """
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
    node_data: NodeDataArray,
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


@nb.njit()
def query_radius(
    X: FloatArray,
    r: float,
    max_count: int,
    # ----- tree data below
    data: FloatArray,
    idx_array: IntArray,
    idx_start: IntArray,
    idx_end: IntArray,
    is_leaf: BoolArray,
    node_data: NodeDataArray,
    rdist: RDist,
    min_max_rdist: MinMaxRDist,
):
    n_queries, n_features = X.shape
    n_data, n_features_ = data.shape
    assert n_features == n_features_
    shape = (n_queries, max_count)
    dists = np.full(shape, np.inf, dtype=data.dtype)
    indices = np.full(shape, n_data, dtype=idx_array.dtype)
    counts = np.empty((n_queries,), dtype=idx_array.dtype)
    query_radius_prealloc(
        X,
        r,
        dists,
        indices,
        counts,
        data,
        idx_array,
        idx_start,
        idx_end,
        is_leaf,
        node_data,
        rdist,
        min_max_rdist,
    )
    return QueryResult(dists, indices, counts)


@nb.njit(parallel=PARALLEL)
def query_radius_prealloc(
    X: FloatArray,
    r: float,
    # ----- preallocated data below
    dists: FloatArray,
    indices: IntArray,
    counts: IntArray,
    # ----- tree data below
    data: FloatArray,
    idx_array: IntArray,
    idx_start: IntArray,
    idx_end: IntArray,
    is_leaf: BoolArray,
    node_data: NodeDataArray,
    rdist: RDist,
    min_max_rdist: MinMaxRDist,
) -> None:
    """
    Perform ball search saving data into preallocated arrays.

    Args:
        X: [n_queries, n_features] float array of query points.
        r: reduced radius (e.g. squared radius values if rdist is norm^2)
            of search.
        dists: [n_queries, max_neighbors] float array into which resulting
            reduced distances are saved.
        indices: [n_queries, max_neighbors] int array into which resulting
            indices are saved.
        counts: [n_queries] int array into which resultin counts of neighbors
            are saved.
        *tree_data: data associated with the BinaryTree.
    """
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
    node_data: NodeDataArray,
    rdist: RDist,
    min_max_rdist: MinMaxRDist,
) -> int:
    if count >= max_count:
        return count

    rdist_LB, rdist_UB = min_max_rdist(node_data[i_node], x)

    # ------------------------------------------------------------
    # Case 1: all node points are outside distance r.
    #         prune this branch.
    if rdist_LB > r:
        pass

    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
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
        ("n_data", INT_TYPE_T),
        ("n_features", INT_TYPE_T),
        ("leaf_size", INT_TYPE_T),
        ("n_nodes", INT_TYPE_T),
        ("data", float_type_t[:, :]),
        ("idx_array", int_type_t[::1]),
        ("idx_start", int_type_t[::1]),
        ("idx_end", int_type_t[::1]),
        ("is_leaf", bool_type_t[::1]),
    ]


class BinaryTree:
    """
    Base class for binary trees.

    This is designed to be extended by jitted classes. To enable this and ensure
    good performance, a number of things are non-standard.

    1. __init__ work is done in _init. This allows derived classes to call
        self._init(*args) rather than forcing them to use super(...).__init__()
        (super calls aren't supported as far as I can tell).
    2. `rdist` and `min_max_rdist` are conceptually functions which would
        ordinarily be implemented as class methods and then passed into
        jitted implementation functions. This forces the object itself to be
        passed into those functions, which results in very slow performance.
        Instead, we implement `rdist` and `min_max_rdist` as properties which
        return `njit`ed functions.

    Derived classes should implement:
    - rdist: property that returns a function that gives the reduced distance
        between two points. reduced distances are distances which preserve
        order with distance but may be easier to compute. For example,
        squared distance is a good reduced distance for normal distance since
        it avoids the need to evaluate the square root.
    - min_max_rdist: function that returns a lower and upper bound on the
        reduced distance between a node and a given point given the `node_data`
        associated with the given node.

    See kd_tree.KDTree for implementation.
    """

    def __init__(
        self,
        data: FloatArray,
        leaf_size: int,
        idx_array: IntArray,
        idx_start: IntArray,
        idx_end: IntArray,
        is_leaf: BoolArray,
        node_data: NodeDataArray,
    ):
        fill_tree_data(data, leaf_size, idx_array, idx_start, idx_end, is_leaf)
        self.data = data
        self.idx_array = idx_array
        self.idx_start = idx_start
        self.idx_end = idx_end
        self.is_leaf = is_leaf
        self.node_data = node_data
        self.n_data, self.n_features = data.shape
        self.n_nodes = len(self.idx_start)
        self._fill_node_data()

    def get_tree_data(self) -> Tuple:
        return (
            self.data,
            self.idx_array,
            self.idx_start,
            self.idx_end,
            self.is_leaf,
            self.node_data,
        )

    def _fill_node_data(self):
        pass

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
        """
        rdist function.

        By making this a property with a callable value, passing this into
        jitted functions does not result in a massive slow down like bound
        member functions does.
        """
        raise NotImplementedError("Abstract method")

    @property
    def min_max_rdist(self) -> MinMaxRDist:
        """
        min_max_rdist function.

        By making this a property with a callable value, passing this into
        jitted functions does not result in a massive slow down like bound
        member functions does.
        """
        raise NotImplementedError("Abstract method")

    def query_radius_prealloc(
        self,
        X: np.ndarray,
        r: float,
        dists: np.ndarray,
        indices: np.ndarray,
        counts: np.ndarray,
    ) -> None:
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

    def query_radius(self, X: np.ndarray, r: float, max_count: int) -> QueryResult:
        """
        Perform ball search on query points X.

        Args:
            X: [n_queries, n_features] float array of query points.
            r: reduced radius of search. Note this may be a squared distance
                depending on rdist/min_max_rdist implementation.
            max_count: maximum number of neighbors to consider. If this number
                of neighbors is found we return, and the returned neighbors
                will not necessarily be the closest neighbors (though they will
                all be within `r` as measured by `rdist`).

        Returns:
            QueryResult: (dists, indices, counts)
        """
        return query_radius(
            X,
            r,
            max_count,
            data=self.data,
            idx_array=self.idx_array,
            idx_start=self.idx_start,
            idx_end=self.idx_end,
            is_leaf=self.is_leaf,
            node_data=self.node_data,
            rdist=self.rdist,
            min_max_rdist=self.min_max_rdist,
        )

    def query_radius_bottom_up_prealloc(
        self,
        X: FloatArray,
        r: float,
        start_nodes: IntArray,
        dists: FloatArray,
        indices: IntArray,
        counts: IntArray,
    ) -> None:
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

    def query_radius_bottom_up(
        self, X: FloatArray, r: float, start_nodes: IntArray, max_count: int
    ) -> QueryResult:
        return query_radius_bottom_up(
            X,
            r,
            start_nodes,
            max_count,
            data=self.data,
            idx_array=self.idx_array,
            idx_start=self.idx_start,
            idx_end=self.idx_end,
            is_leaf=self.is_leaf,
            node_data=self.node_data,
            rdist=self.rdist,
            min_max_rdist=self.min_max_rdist,
        )

    def rejection_sample_query_prealloc(
        self,
        rejection_r: float,
        query_r: float,
        start_nodes: IntArray,
        sample_indices: IntArray,
        dists: FloatArray,
        query_indices: IntArray,
        counts: IntArray,
        consumed: BoolArray,
    ):
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

    def rejection_sample_query(
        self,
        rejection_r,
        query_r,
        start_nodes: IntArray,
        max_samples: int,
        max_counts: int,
    ) -> RejectionSampleQueryResult:
        return rejection_sample_query(
            rejection_r,
            query_r,
            start_nodes,
            max_samples,
            max_counts,
            data=self.data,
            idx_array=self.idx_array,
            idx_start=self.idx_start,
            idx_end=self.idx_end,
            is_leaf=self.is_leaf,
            node_data=self.node_data,
            rdist=self.rdist,
            min_max_rdist=self.min_max_rdist,
        )

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
        min_dists: FloatArray,  # in_size, minimum distances
        heap: ih.IndexHeap,  # assumed to be heapified
    ) -> float:
        return ifp_sample_query_prealloc(
            query_r=query_r,
            start_nodes=start_nodes,
            sample_indices=sample_indices,
            dists=dists,
            query_indices=query_indices,
            counts=counts,
            min_dists=min_dists,
            heap=heap,
            data=self.data,
            idx_array=self.idx_array,
            idx_start=self.idx_start,
            idx_end=self.idx_end,
            is_leaf=self.is_leaf,
            node_data=self.node_data,
            rdist=self.rdist,
            min_max_rdist=self.min_max_rdist,
        )

    def ifp_sample_query(
        self,
        query_r: float,
        start_nodes: IntArray,
        sample_size: int,
        max_counts: int,
        eps: float = 1e-8,
    ) -> IFPSampleQueryResult:
        return ifp_sample_query(
            query_r,
            start_nodes,
            sample_size,
            max_counts,
            eps=eps,
            data=self.data,
            idx_array=self.idx_array,
            idx_start=self.idx_start,
            idx_end=self.idx_end,
            is_leaf=self.is_leaf,
            node_data=self.node_data,
            rdist=self.rdist,
            min_max_rdist=self.min_max_rdist,
        )

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
        min_dists: FloatArray,
        heap_priorities: FloatArray,
        heap_indices: IntArray,
    ) -> float:
        """
        Simultaneous sampling and querying with preallocated data.

        Returns minimum reduced distance of final sampled point. All points
        should be within this reduced distance of a sampled point.
        """
        return rejection_ifp_sample_query_prealloc(
            rejection_r=rejection_r,
            query_r=query_r,
            start_nodes=start_nodes,
            sample_indices=sample_indices,
            dists=dists,
            query_indices=query_indices,
            counts=counts,
            consumed=consumed,
            min_dists=min_dists,
            heap_priorities=heap_priorities,
            heap_indices=heap_indices,
            data=self.data,
            idx_array=self.idx_array,
            idx_start=self.idx_start,
            idx_end=self.idx_end,
            is_leaf=self.is_leaf,
            node_data=self.node_data,
            rdist=self.rdist,
            min_max_rdist=self.min_max_rdist,
        )

    def rejection_ifp_sample_query(
        self,
        rejection_r: float,
        query_r: float,
        start_nodes: IntArray,
        sample_size: int,
        max_counts: int,
    ) -> IFPSampleQueryResult:
        """
        Perform simultaneous rejection_ifp sampling and querying.

        Args:
            rejection_r: reduced radius of rejections.
            query_r: reduced radius of queries.
            start_nodes: [in_size] leaf indices of points in data, e.g.
                from `self.get_node_indices()`.

        Returns:
            IFPSampleQueryResult:
                IFPSampleResult:
                    - indices
                    - min_dists
                    - min_dist

                QueryResult:
                    - dists
                    - indices
                    - counts
        """
        return rejection_ifp_sample_query(
            rejection_r=rejection_r,
            query_r=query_r,
            start_nodes=start_nodes,
            sample_size=sample_size,
            max_counts=max_counts,
            data=self.data,
            idx_array=self.idx_array,
            idx_start=self.idx_start,
            idx_end=self.idx_end,
            is_leaf=self.is_leaf,
            node_data=self.node_data,
            rdist=self.rdist,
            min_max_rdist=self.min_max_rdist,
        )

    def get_node_indices_prealloc(self, node_indices: IntArray):
        get_node_indices_prealloc(
            self.idx_array, self.idx_start, self.idx_end, self.is_leaf, node_indices
        )

    def get_node_indices(self):
        """Get the node indices of the leaf containing each point in data."""
        return get_node_indices(
            idx_array=self.idx_array,
            idx_start=self.idx_start,
            idx_end=self.idx_end,
            is_leaf=self.is_leaf,
        )

    def permute(self, perm):
        """
        Permute this tree's `data` array and modify `idx_array`.

        This is for when you want returned indices from e.g. queries or
        samplings to give indices into a permuted data but don't want to
        rebuild the tree.
        """
        self.data, self.idx_array = permute_tree(self.data, self.idx_array, perm)


def binary_tree(
    data: FloatArray, leaf_size: int = 40, int_type=INT_TYPE, bool_type=BOOL_TYPE
):
    return BinaryTree(
        data,
        leaf_size,
        *create_tree_data(data, leaf_size, int_type, bool_type),
        node_data=None
    )
