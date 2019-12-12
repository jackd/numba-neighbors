from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
from numba_neighbors import kdtree as kd
from numba_neighbors.binary_tree import simultaneous_sort_partial
from sklearn.neighbors import KDTree as sk_KDTree


class KDTreeTest(unittest.TestCase):

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
        D = 3
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

        numba_tree = kd.get_tree_data(data, leaf_size=leaf_size)
        dists = np.full((n, max_neighbors), np.inf, dtype=np.float32)
        indices = np.zeros((n, max_neighbors), dtype=np.int64)
        counts = np.zeros((n,), dtype=np.int64)

        kd.query_radius_prealloc(
            X,
            r2,
            dists,
            indices,
            counts,
            n_samples=numba_tree.n_samples,
            n_features=numba_tree.n_features,
            leaf_size=numba_tree.leaf_size,
            n_levels=numba_tree.n_levels,
            n_nodes=numba_tree.n_nodes,
            data=numba_tree.data,
            idx_array=numba_tree.idx_array,
            idx_start=numba_tree.idx_start,
            idx_end=numba_tree.idx_end,
            is_leaf=numba_tree.is_leaf,
            node_lower_bounds=numba_tree.node_lower_bounds,
            node_upper_bounds=numba_tree.node_upper_bounds,
        )

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

        numba_tree = kd.get_tree_data(data, leaf_size=leaf_size)

        dists = np.full((n, max_neighbors), np.inf, dtype=np.float32)
        indices = np.zeros((n, max_neighbors), dtype=np.int64)
        counts = np.zeros((n,), dtype=np.int64)
        start_nodes = np.zeros((n,), dtype=np.int64)

        # nodes = np.full((N,), (-1,), dtype=np.int64)
        # idx_array = numba_tree.idx_array
        # for i in range(numba_tree.n_nodes):
        #     if numba_tree.is_leaf[i]:
        #         nodes[idx_array[numba_tree.idx_start[i]:numba_tree.
        #                         idx_end[i]]] = i
        # start_nodes = nodes[X_indices]
        start_nodes = kd.get_node_indices(
            n_samples=numba_tree.n_samples,
            n_nodes=numba_tree.n_nodes,
            idx_array=numba_tree.idx_array,
            idx_start=numba_tree.idx_start,
            idx_end=numba_tree.idx_end,
            is_leaf=numba_tree.is_leaf,
        )[X_indices]

        kd.query_radius_bottom_up_prealloc(
            X,
            r2,
            start_nodes,
            dists,
            indices,
            counts,
            n_samples=numba_tree.n_samples,
            n_features=numba_tree.n_features,
            leaf_size=numba_tree.leaf_size,
            n_levels=numba_tree.n_levels,
            n_nodes=numba_tree.n_nodes,
            data=numba_tree.data,
            idx_array=numba_tree.idx_array,
            idx_start=numba_tree.idx_start,
            idx_end=numba_tree.idx_end,
            is_leaf=numba_tree.is_leaf,
            node_lower_bounds=numba_tree.node_lower_bounds,
            node_upper_bounds=numba_tree.node_upper_bounds,
        )

        simultaneous_sort_partial(dists, indices, counts)
        mask = np.tile(np.expand_dims(np.arange(max_neighbors), 0),
                       (n, 1)) < np.expand_dims(counts, axis=1)
        flat_dists = dists[mask]
        flat_indices = indices[mask]

        np.testing.assert_equal(np.sum(counts), np.sum(expected_counts))
        np.testing.assert_equal(flat_indices, expected_indices)
        np.testing.assert_allclose(np.sqrt(flat_dists), expected_dists)


if __name__ == '__main__':
    unittest.main()
