import numpy as np

from numba_neighbors.index_heap import padded_index_heap

heap = padded_index_heap(np.zeros((10,), dtype=np.float32), np.arange(10), 20)
print(heap.pop())
print(heap.pop())
print(heap.pop())
print(heap.pop())
