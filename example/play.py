from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numba_neighbors import kdtree as kd
from numba_neighbors.binary_tree import simultaneous_sort_partial
from sklearn.neighbors import KDTree as sk_KDTree

np.random.seed(124)
N = 64
n = 1
D = 3
r = 0.1
max_neighbors = 32
leaf_size = 16
data = np.random.uniform(size=(N, D)).astype(np.float32)
X_indices = [23]
X = data[X_indices]

for i, d in enumerate(data):
    print(i, d)

sk_tree = sk_KDTree(data, leaf_size=leaf_size)

expected_indices, expected_dists = sk_tree.query_radius(X,
                                                        r,
                                                        return_distance=True,
                                                        sort_results=True)
expected_counts = [d.size for d in expected_dists]
expected_dists = np.concatenate(expected_dists, axis=0)
expected_indices = np.concatenate(expected_indices, axis=0)

numba_tree = kd.get_tree_data(data, leaf_size=leaf_size)

# data = data[numba_tree.idx_array]
# numba_tree = kd.get_tree_data(data, leaf_size=leaf_size)
# X = data[X_indices]
# print(nt2.idx_array)
# return

dists = np.full((n, max_neighbors), np.inf, dtype=np.float32)
indices = np.zeros((n, max_neighbors), dtype=np.int64)
counts = np.zeros((n,), dtype=np.int64)

nodes = np.full((N,), (-1,), dtype=np.int64)
idx_array = numba_tree.idx_array

xi = X_indices[0]
for i in range(numba_tree.n_nodes):
    lower = numba_tree.idx_start[i]
    upper = numba_tree.idx_end[i]
    # if lower <= ixi < upper:
    #     print(i, numba_tree.node_bounds[:, i])
    if numba_tree.is_leaf[i]:

        nodes[numba_tree.idx_array[lower:upper]] = i
print(numba_tree.idx_start)
print(numba_tree.idx_end)
print(nodes)
ixi = nodes[X_indices[0]]

print(idx_array)
print(xi, ixi)
assert (np.all(nodes >= 0))
# start_nodes = nodes[idx_array[X_indices]]
print(nodes[ixi])

sn = nodes[ixi]
si = numba_tree.idx_start[sn]
se = numba_tree.idx_end[sn]
idx = numba_tree.idx_array[si:se]
print('---')
print(sn, si, se)
print('---')
print(numba_tree.node_bounds[:, sn])
print('---')
print(data[xi])
# print(idx_array[numba_tree.idx_start[]])

# kd.query_radius_bottom_up_prealloc(numba_tree, X, r, dists, indices,
#                                    counts, start_nodes)

# simultaneous_sort_partial(dists, indices, counts)
# mask = np.tile(np.expand_dims(np.arange(max_neighbors), 0),
#                (n, 1)) < np.expand_dims(counts, axis=1)
# flat_dists = dists[mask]
# flat_indices = indices[mask]

# np.testing.assert_equal(np.sum(counts), np.sum(expected_counts))
# np.testing.assert_equal(flat_indices, expected_indices)
# np.testing.assert_allclose(np.sqrt(flat_dists), expected_dists)
