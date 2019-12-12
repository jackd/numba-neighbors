from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['NUMBA_DISABLE_JIT'] = '1'
import numpy as np
from numba_neighbors import kdtree as kd
import matplotlib.pyplot as plt

N = 1024
n = 256
D = 2
# rejection_r = 0.1
query_r = 0.1
max_neighbors = 64
leaf_size = 16

np.random.seed(124)
data = np.random.uniform(size=(N, D)).astype(kd.FLOAT_TYPE)

tree = kd.KDTree(data, leaf_size=leaf_size)
sample_result, query_result = tree.rejection_ifp_sample_query(
    query_r**2, tree.get_node_indices(), n, max_neighbors)
print(np.max(query_result.counts))
print(sample_result.min_dist, np.max(sample_result.min_dists))


def vis(x0,
        sample_indices,
        small_balls=True,
        big_balls=False,
        labels=False,
        aspect=1):
    x1 = x0[sample_indices]
    x0 = x0.T
    x1 = x1.T

    plt.axes([0, 0, 1, 1])
    ax = plt.gca()
    for x in x1.T:
        if small_balls:
            ax.add_patch(
                plt.Circle(x,
                           radius=query_r / 2,
                           alpha=0.4,
                           fill=1,
                           color='red'))
        if big_balls:
            ax.add_patch(
                plt.Circle(x, radius=query_r, alpha=0.15, fill=1, color='red'))

    if labels:
        for i, xi in enumerate(x0.T):
            ax.annotate(str(i), xi)

    ax.scatter(*x0, c='blue', alpha=0.5, s=10)
    ax.scatter(*x1, c='red', alpha=1, s=10)
    ax.axis('off')
    ax.set_aspect(1)


vis(data, sample_result.indices)
plt.show()
