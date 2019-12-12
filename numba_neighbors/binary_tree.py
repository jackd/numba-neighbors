from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import numba as nb
import numpy as np

INT_TYPE = np.int64
INT_TYPE_T = nb.int64

FLOAT_TYPE = np.float32
FLOAT_TYPE_T = nb.float32

BOOL_TYPE = np.uint8
BOOL_TYPE_T = nb.uint8

IntArray = np.ndarray
FloatArray = np.ndarray


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
