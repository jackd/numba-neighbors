from numba_neighbors.binary_tree import simultaneous_sort

import numpy as np
import unittest
from numba_neighbors import binary_tree as bt


class BinaryTreeTest(unittest.TestCase):

    def test_build(self):
        np.random.seed(123)
        N = 256
        data = np.random.random(size=(N, 1)).astype(np.float32)
        tree = bt.binary_tree(data, leaf_size=1)
        np.testing.assert_equal(data[tree.idx_array], np.sort(data, axis=0))

    def test_simultaneous_sort(self):
        np.random.seed(123)
        N = 8
        k = 1000
        dist = np.random.uniform(size=(N, k), high=100).astype(np.float32)
        idx = np.random.uniform(size=(N, k), high=1000).astype(np.int64)

        i = np.argsort(dist)
        expected_dist = np.take_along_axis(dist, i, axis=1)
        expected_idx = np.take_along_axis(idx, i, axis=1)

        simultaneous_sort(dist, idx)
        np.testing.assert_allclose(dist, expected_dist)
        np.testing.assert_equal(idx, expected_idx)

    def test_permute_tree(self):
        N = 1024
        data = np.random.uniform(size=(N, 3))

        idx_array = np.arange(N)
        np.random.shuffle(idx_array)

        perm = np.arange(N)
        np.random.shuffle(perm)

        permuted_data, permuted_idx_array = bt.permute_tree(
            data, idx_array, perm)

        actual = permuted_data[permuted_idx_array]
        expected = data[idx_array]
        np.testing.assert_equal(actual, expected)

    def rejection_sample_valid(self):
        np.random.seed(123)
        N = 256
        data = np.random.random(size=(N, 1)).astype(np.float32)
        tree = bt.binary_tree(data, leaf_size=16)
        r0 = 0.2
        r1 = 0.1

        dists, indices, counts = tree.query_radius(data, r=r0**2, max_count=N)

        valid = dists < r1**2
        actual_sample_indices, actual_count = bt.rejection_sample_precomputed(
            indices, counts, N, valid=valid)

        dists, indices, counts = tree.query_radius(data, r=r1**2, max_count=N)
        expected_sample_indices, expected_count = bt.rejection_sample_precomputed(
            indices, counts, N)

        np.testing.assert_equal(actual_sample_indices, expected_sample_indices)
        np.testing.assert_equal(actual_count, expected_count)


if __name__ == '__main__':
    unittest.main()
