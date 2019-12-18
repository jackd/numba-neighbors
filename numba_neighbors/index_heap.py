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
        _heapify(self)

    def pop(self):
        return _heappop(self)

    def push(self, priority, value):
        return _heappush(self, priority, value)


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
def _getitem(self, index):
    # assert (0 <= index)
    # assert (index < self.length)
    return self.priorities[index], self.indices[index]


@njit(inline='always')
def _setitem(self, index, item):
    pr, val = item
    self.priorities[index] = pr
    self.indices[index] = val


@njit(inline='always')
def _replaceitem(self, dst_index, src_index):
    """Equivalent to self._setitem(dst_index, self._getitem(src_index))."""
    self.priorities[dst_index] = self.priorities[src_index]
    self.indices[dst_index] = self.indices[src_index]


@njit(inline='always')
def _arr_append(self, priority, value):
    # assert (self.length < self.max_length)
    self.priorities[self.length] = priority
    self.indices[self.length] = value
    self.length += 1


@njit(inline='always')
def _arr_pop(self) -> Tuple:
    # assert (self.length > 0)
    self.length -= 1
    priority = self.priorities[self.length]
    value = self.indices[self.length]
    return priority, value


@njit(inline='always')
def _less(self, i1: int, i2: int) -> bool:
    left = self.priorities[i1]
    right = self.priorities[i2]
    if left == right:
        return self.indices[i1] < self.indices[i2]
    else:
        return left < right


@njit(inline='always')
def _siftup(self, pos):
    endpos = self.length
    startpos = pos
    newitem = _getitem(self, pos)
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2 * pos + 1  # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not _less(self, childpos, rightpos):
            childpos = rightpos
        # Move the smaller child up.
        _replaceitem(self, pos, childpos)
        pos = childpos
        childpos = 2 * pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    _setitem(self, pos, newitem)
    _siftdown(self, startpos, pos)


@njit(inline='always')
def _siftdown(self, startpos, pos):
    newpr, newind = _getitem(self, pos)
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parentpr = self.priorities[parentpos]
        if newpr < parentpr:
            self.priorities[pos] = parentpr
            self.indices[pos] = self.indices[parentpos]
            pos = parentpos
            continue
        break
    self.priorities[pos] = newpr
    self.indices[pos] = newind


@njit(inline='always')
def _heapify(self):
    """Transform list into a heap, in-place, in O(len(x)) time."""
    n = self.length
    # Transform bottom-up.  The largest index there's any point to looking at
    # is the largest with a child index in-range, so must have 2*i + 1 < n,
    # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
    # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
    # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
    for i in range(n // 2 - 1, -1, -1):
        _siftup(self, i)


@njit(inline='always')
def _heappush(self, priority, value):
    """Push item onto heap, maintaining the heap invariant."""
    # if self.length == self.max_length - 1:
    #     # raise ValueError('Cannot push: heap full')
    #     # raise IndexError
    #     # raise Exception
    #     print('Tried pushing onto full heap, oh dear!')
    _arr_append(self, priority, value)
    _siftdown(self, 0, self.length - 1)


@njit(inline='always')
def _heappop(self) -> Tuple:
    """Pop the smallest item off the heap, maintaining the heap invariant."""
    # if self.length == 0:
    #     # raise ValueError('Cannot pop: heap empty')
    #     # raise IndexError
    #     # raise Exception
    #     print('Tried popping off empty heap, oh dear!')
    lastelt = _arr_pop(self)
    if self.length > 0:
        returnitem = _getitem(self, 0)
        _setitem(self, 0, lastelt)
        _siftup(self, 0)
        return returnitem
    return lastelt
