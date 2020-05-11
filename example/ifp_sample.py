import matplotlib.pyplot as plt
import numpy as np

from numba_neighbors import kd_tree as kd

N = 1024
n = 70
D = 2
# rejection_r = 0.1
query_r = 0.1
max_neighbors = 64
leaf_size = 16

r2 = query_r ** 2

np.random.seed(124)
data = np.random.uniform(size=(N, D)).astype(kd.FLOAT_TYPE)

tree = kd.KDTree(data, leaf_size=leaf_size)
sample_result0, query_result0 = tree.ifp_sample_query(
    r2, tree.get_node_indices(), n, max_neighbors
)
sample_result, query_result = tree.rejection_ifp_sample_query(
    r2, r2, tree.get_node_indices(), n, max_neighbors
)


def vis(
    x0, sample_indices, query_result, small_balls=True, big_balls=False, labels=False,
):
    x1 = x0[sample_indices]
    xn = x0[query_result.indices[0, : query_result.counts[0]]]
    x10 = x1[0]
    x0 = x0.T
    x1 = x1.T
    xn = xn.T

    plt.axes([0, 0, 1, 1])
    ax = plt.gca()
    for x in x1.T:
        if small_balls:
            ax.add_patch(
                plt.Circle(x, radius=query_r / 2, alpha=0.4, fill=1, color="red")
            )
        if big_balls:
            ax.add_patch(plt.Circle(x, radius=query_r, alpha=0.15, fill=1, color="red"))

    if labels:
        for i, xi in enumerate(x0.T):
            ax.annotate(str(i), xi)

    ax.scatter(*x0, c="blue", alpha=0.5, s=10)
    ax.scatter(*x1, c="red", alpha=1, s=10)
    ax.scatter(*xn, c="green", alpha=1, s=10)
    ax.scatter(*x10, c="black", alpha=1, s=10)
    ax.add_patch(plt.Circle(x10, radius=query_r, alpha=0.15, fill=1, color="green"))
    ax.axis("off")
    ax.set_aspect(1)


vis(data, sample_result.indices, query_result, big_balls=False)
plt.show()
