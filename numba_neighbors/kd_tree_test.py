from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
# os.environ['NUMBA_DISABLE_JIT'] = '1'

import numpy as np
import unittest
from numba_neighbors import kd_tree as kd
from numba_neighbors.binary_tree import simultaneous_sort_partial
from sklearn.neighbors import KDTree as sk_KDTree


class KDTreeTest(unittest.TestCase):

    def tree(self, data, leaf_size):
        return kd.KDTree(data, leaf_size=leaf_size)

    @property
    def num_dims(self):
        return 3

    # def test_construction_consistent(self):
    #     np.random.seed(123)
    #     N = 1024
    #     D = 3
    #     data = np.random.uniform(size=(N, D)).astype(np.float32)
    #     leaf_size = 16

    #     actual = kd.get_tree_data(data, leaf_size=leaf_size)
    #     expected = sk_KDTree(data, leaf_size=leaf_size)

    #     np.testing.assert_equal(actual.n_features, D)
    #     np.testing.assert_equal(actual.n_samples, N)
    #     np.testing.assert_equal(actual.leaf_size, leaf_size)
    # np.testing.assert_equal(actual.n_nodes, len(expected.node_data))

    # np.testing.assert_equal(actual.idx_array, expected.idx_array)
    # np.testing.assert_allclose(actual.node_bounds, expected.node_bounds)
    # np.testing.assert_equal(actual.idx_start,
    #                         [nd['idx_start'] for nd in expected.node_data])
    # np.testing.assert_equal(actual.idx_end,
    #                         [nd['idx_end'] for nd in expected.node_data])
    # np.testing.assert_equal(actual.is_leaf,
    #                         [nd['is_leaf'] for nd in expected.node_data])
    # np.testing.assert_allclose(actual.radius,
    #                            [nd['radius'] for nd in expected.node_data])

    def test_query_consistent(self):
        np.random.seed(123)
        N = 1024
        n = 256
        D = self.num_dims
        r = 0.05
        r2 = r * r
        max_neighbors = 32
        leaf_size = 16
        data = np.random.uniform(size=(N, D)).astype(np.float32)
        X_indices = np.random.choice(N, size=n, replace=False)
        X = data[X_indices]

        sk_tree = sk_KDTree(data, leaf_size=leaf_size)

        expected_indices, expected_dists = sk_tree.query_radius(
            X, r, return_distance=True, sort_results=True)
        expected_counts = [d.size for d in expected_dists]
        expected_dists = np.concatenate(expected_dists, axis=0)
        expected_indices = np.concatenate(expected_indices, axis=0)

        numba_tree = self.tree(data, leaf_size)

        dists = np.full((n, max_neighbors), np.inf, dtype=np.float32)
        indices = np.zeros((n, max_neighbors), dtype=np.int64)
        counts = np.zeros((n,), dtype=np.int64)

        numba_tree.query_radius_prealloc(X, r2, dists, indices, counts)

        simultaneous_sort_partial(dists, indices, counts)
        mask = np.tile(np.expand_dims(np.arange(max_neighbors), 0),
                       (n, 1)) < np.expand_dims(counts, axis=1)
        flat_dists = dists[mask]
        flat_indices = indices[mask]

        np.testing.assert_equal(np.sum(counts), np.sum(expected_counts))
        np.testing.assert_equal(flat_indices, expected_indices)
        np.testing.assert_allclose(np.sqrt(flat_dists), expected_dists)

    def test_query_bottom_up_consistent(self):
        np.random.seed(124)
        N = 1024
        n = 256
        D = 3
        r = 0.1
        r2 = r * r
        max_neighbors = 32
        leaf_size = 16
        data = np.random.uniform(size=(N, D)).astype(np.float32)
        X_indices = np.random.choice(N, size=n, replace=False)
        X = data[X_indices]

        sk_tree = sk_KDTree(data, leaf_size=leaf_size)

        expected_indices, expected_dists = sk_tree.query_radius(
            X, r, return_distance=True, sort_results=True)
        expected_counts = [d.size for d in expected_dists]
        expected_dists = np.concatenate(expected_dists, axis=0)
        expected_indices = np.concatenate(expected_indices, axis=0)

        numba_tree = self.tree(data, leaf_size=leaf_size)

        dists = np.full((n, max_neighbors), np.inf, dtype=np.float32)
        indices = np.zeros((n, max_neighbors), dtype=np.int64)
        counts = np.zeros((n,), dtype=np.int64)
        start_nodes = np.zeros((n,), dtype=np.int64)

        start_nodes = numba_tree.get_node_indices()[X_indices]

        numba_tree.query_radius_bottom_up_prealloc(X, r2, start_nodes, dists,
                                                   indices, counts)

        simultaneous_sort_partial(dists, indices, counts)
        mask = np.tile(np.expand_dims(np.arange(max_neighbors), 0),
                       (n, 1)) < np.expand_dims(counts, axis=1)
        flat_dists = dists[mask]
        flat_indices = indices[mask]

        np.testing.assert_equal(np.sum(counts), np.sum(expected_counts))
        np.testing.assert_equal(flat_indices, expected_indices)
        np.testing.assert_allclose(np.sqrt(flat_dists), expected_dists)


class KDTree3Test(KDTreeTest):

    @property
    def num_dims(self):
        return 3

    def tree(self, data, leaf_size):
        return kd.KDTree3(data, leaf_size=leaf_size)


if __name__ == '__main__':
    unittest.main()
