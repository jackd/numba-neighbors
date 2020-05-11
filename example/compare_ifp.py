import os

import numpy as np

from numba_neighbors import binary_tree as bt
from numba_neighbors import kd_tree as kd

os.environ["NUMBA_DISABLE_JIT"] = "1"


N = 100
n = 50
D = 1
# rejection_r = 0.1
query_r = 0.3
max_neighbors = 100
leaf_size = 16

r2 = query_r ** 2

np.random.seed(124)
data = np.random.uniform(size=(N, D)).astype(kd.FLOAT_TYPE)
# data.sort(axis=0)
print(data)

tree = kd.KDTree(data, leaf_size=leaf_size)

qr = tree.query_radius_bottom_up(data, r2, tree.get_node_indices(), max_neighbors)

sr_rej = bt.rejection_ifp_sample_precomputed(qr.dists, qr.indices, qr.counts, n)
print(sr_rej.indices)

sr = bt.ifp_sample_precomputed(qr.dists, qr.indices, qr.counts, n)
print(sr.indices)
