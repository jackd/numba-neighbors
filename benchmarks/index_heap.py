from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import heapq
from numba_neighbors.index_heap import IndexHeap
from numba_neighbors.index_heap2 import IndexHeap as IndexHeap2
from numba_neighbors.index_heap import padded_index_heap
from numba_neighbors.benchmark_utils import BenchmarkManager
from numba import njit
import numpy as np

length = 1024
max_length = 32 * length

heapify_bm = BenchmarkManager()


@heapify_bm.benchmark('heapq')
def heapq_impl():
    np.random.seed(123)
    priorities = np.random.random(size=(length,)).astype(np.float32)
    indices = np.arange(length)
    heap = list(zip(priorities, indices))
    heapq.heapify(heap)


@heapify_bm.benchmark('index_heap')
@njit()
def index_heap_impl():
    np.random.seed(123)
    priorities = np.random.random(size=(length,)).astype(np.float32)
    indices = np.arange(length)
    # iheap = padded_index_heap(priorities, indices, length)
    # iheap = padded_index_heap(priorities, indices, max_length)
    iheap = IndexHeap(priorities, indices, length)
    iheap.heapify()


@heapify_bm.benchmark('index_heap2')
@njit()
def index_heap2_impl():
    np.random.seed(123)
    priorities = np.random.random(size=(length,)).astype(np.float32)
    indices = np.arange(length)
    # iheap = padded_index_heap(priorities, indices, length)
    # iheap = padded_index_heap(priorities, indices, max_length)
    iheap = IndexHeap2(priorities, indices, length)
    iheap.heapify()


heapify_bm.run_benchmarks(20, 100)
