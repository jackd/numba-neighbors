from typing import Tuple, NamedTuple
import numpy as np
from numba import jitclass, njit
from numba import types


@jitclass([('priorities', types.float32[:]), ('indices', types.int64[:]),
           ('max_length', types.int64), ('length', types.int64)])
class IndexHeap(object):

    def __init__(self,
                 priorities: np.ndarray,
                 indices: np.ndarray,
                 length: int = 0):
        self.priorities = priorities
        self.indices = indices
        self.length = length
        self.max_length = priorities.size

    def __len__(self):
        return self.length

    def heapify(self):
        _heapify(self.priorities, self.indices, self.length)

    def pop(self):
        item, self.length = _heappop(self.priorities, self.indices, self.length)
        return item

    def push(self, priority, index):
        self.length = _heappush(priority, index, self.priorities, self.indices,
                                self.length)


@njit()
def padded_index_heap(priorities, indices, max_length):
    length = priorities.size
    assert (indices.size == length)
    actual_priorities = np.empty((max_length,), dtype=priorities.dtype)
    actual_priorities[:length] = priorities
    actual_indices = np.empty((max_length,), dtype=indices.dtype)
    actual_indices[:length] = indices
    return IndexHeap(actual_priorities, actual_indices, length)


@njit(inline='always')
def _getitem(index, priorities, indices):
    # assert (0 <= index)
    # assert (index < self.length)
    return priorities[index], indices[index]


@njit(inline='always')
def _setitem(index, item, priorities, indices):
    pr, val = item
    priorities[index] = pr
    indices[index] = val


@njit(inline='always')
def _replaceitem(dst_index, src_index, priorities, indices):
    """Equivalent to self._setitem(dst_index, self._getitem(src_index))."""
    priorities[dst_index] = priorities[src_index]
    indices[dst_index] = indices[src_index]


@njit(inline='always')
def _arr_append(priority, value, priorities, indices, length) -> int:
    # assert (self.length < self.max_length)
    priorities[length] = priority
    indices[length] = value
    return length + 1


@njit(inline='always')
def _arr_pop(priorities, indices, length) -> Tuple[Tuple, int]:
    # assert (self.length > 0)
    length = length - 1
    priority = priorities[length]
    value = indices[length]
    return (priority, value), length


@njit(inline='always')
def _siftup(pos, priorities, indices, length):
    endpos = length
    startpos = pos
    newitem = _getitem(pos, priorities, indices)
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2 * pos + 1  # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not priorities[childpos] < priorities[rightpos]:
            childpos = rightpos
        # Move the smaller child up.
        _replaceitem(pos, childpos, priorities, indices)
        pos = childpos
        childpos = 2 * pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    _setitem(pos, newitem, priorities, indices)
    _siftdown(startpos, pos, priorities, indices)


@njit(inline='always')
def _siftdown(startpos, pos, priorities, indices):
    newpr, newind = _getitem(pos, priorities, indices)
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parentpr = priorities[parentpos]
        if newpr < parentpr:
            priorities[pos] = parentpr
            indices[pos] = indices[parentpos]
            pos = parentpos
            continue
        break
    priorities[pos] = newpr
    indices[pos] = newind


@njit(inline='always')
def _heapify(priorities, indices, length):
    """Transform list into a heap, in-place, in O(len(x)) time."""
    # Transform bottom-up.  The largest index there's any point to looking at
    # is the largest with a child index in-range, so must have 2*i + 1 < n,
    # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
    # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
    # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
    for i in range(length // 2 - 1, -1, -1):
        _siftup(i, priorities, indices, length)


@njit(inline='always')
def _heappush(priority, value, priorities, indices, length):
    """Push item onto heap, maintaining the heap invariant."""
    length = _arr_append(priority, value, priorities, indices, length)
    _siftdown(0, length - 1, priorities, indices)
    return length


@njit(inline='always')
def _heappop(priorities, indices, length) -> Tuple:
    """Pop the smallest item off the heap, maintaining the heap invariant."""
    lastelt, length = _arr_pop(priorities, indices, length)  # pylint: disable=unbalanced-tuple-unpacking
    if length > 0:
        returnitem = _getitem(0, priorities, indices)
        _setitem(0, lastelt, priorities, indices)
        _siftup(0, priorities, indices, length)
        return returnitem, length
    return lastelt, length
