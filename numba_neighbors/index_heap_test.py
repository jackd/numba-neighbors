import heapq
import os
import unittest

import numpy as np

from numba_neighbors import index_heap as ih
from numba_neighbors import index_heap2 as ih2

os.environ["NUMBA_DISABLE_JIT"] = "1"


max_length = 100
length = 50


class IndexHeapTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.priorities = np.random.uniform(size=(max_length,)).astype(np.float32)
        self.indices = np.arange(max_length)
        self.heap = list(zip(self.priorities[:length], self.indices[:length]))
        heapq.heapify(self.heap)

        self.iheap = ih.IndexHeap(self.priorities, self.indices, length)
        self.iheap.heapify()

    def _assert_heaps_equal(self):
        length = len(self.heap)
        priorities, indices = zip(*self.heap)
        self.assertEqual(self.iheap.length, length)
        np.testing.assert_equal(self.iheap.priorities[:length], priorities)
        np.testing.assert_equal(self.iheap.indices[:length], indices)

    def test_heapify(self):
        self._assert_heaps_equal()

    def test_heappop(self):
        expectedpr, expectedval = heapq.heappop(self.heap)
        actualpr, actualval = self.iheap.pop()
        self.assertEqual(actualpr, expectedpr)
        self.assertEqual(actualval, expectedval)

    def test_heappush(self):
        heapq.heappush(self.heap, (self.priorities[length], self.indices[length]))
        self.iheap.push(self.priorities[length], self.indices[length])
        self._assert_heaps_equal()


class IndexHeap2Test(IndexHeapTest):
    def setUp(self):
        np.random.seed(123)
        self.priorities = np.random.uniform(size=(max_length,)).astype(np.float32)
        self.indices = np.arange(max_length)
        self.heap = list(zip(self.priorities[:length], self.indices[:length]))
        heapq.heapify(self.heap)

        self.iheap = ih2.IndexHeap(self.priorities, self.indices, length)
        self.iheap.heapify()


if __name__ == "__main__":
    unittest.main()
