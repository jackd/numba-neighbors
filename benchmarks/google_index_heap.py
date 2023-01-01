import heapq

import numpy as np
from numba import njit

import google_benchmark as benchmark
from numba_neighbors.index_heap import IndexHeap
from numba_neighbors.index_heap2 import IndexHeap as IndexHeap2

length = 1024
max_length = 32 * length


@njit()
def get_inputs():
    np.random.seed(123)
    priorities = np.random.random(size=(length,)).astype(np.float32)
    indices = np.arange(length)
    return priorities, indices


@benchmark.register
def heapq_impl(state):
    def fn():
        priorities, indices = get_inputs()
        heap = list(zip(priorities, indices))
        return heapq.heapify(heap)

    fn()
    while state:
        fn()


@benchmark.register
def index_heap_impl(state):
    @njit()
    def fn():
        priorities, indices = get_inputs()
        iheap = IndexHeap(priorities, indices, length)
        return iheap.heapify()

    fn()
    while state:
        fn()


@benchmark.register
def index_heap2_impl(state):
    @njit()
    def fn():
        priorities, indices = get_inputs()
        iheap = IndexHeap2(priorities, indices, length)
        return iheap.heapify()

    fn()
    while state:
        fn()


if __name__ == "__main__":
    benchmark.main()
