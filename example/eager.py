import numpy as np
from numba import float32, float64, int64, njit, prange, void

from numba_neighbors.binary_tree import PARALLEL
from numba_neighbors.kd_tree import KDTree3


@njit(
    void(
        float32[:, :],
        int64,
        int64,
        float64,
        int64,
        int64,
        int64,
        int64[:],
        float32[:, :],
        int64[:, :],
        int64[:],
    ),
    parallel=PARALLEL,
    fastmath=True,
)
def ifp_sample_query_prealloc(
    coords: np.ndarray,
    valid_size: int,
    out_size: int,
    radius: float,
    k_query: int,
    k_return: int,
    leaf_size: int,
    sample_indices,
    dists,
    neigh_indices,
    row_lengths,
):
    r2 = radius * radius
    coords = coords[:valid_size]
    tree = KDTree3(coords, leaf_size)
    start_nodes = tree.get_node_indices()

    if valid_size <= out_size:
        # we can skip sampling and just query
        # preallocating according to the output size is faster than querying
        # and then padding.
        for i in prange(valid_size):  # pylint: disable=not-an-iterable
            sample_indices[i] = i
        for i in prange(valid_size, out_size):  # pylint: disable=not-an-iterable
            sample_indices[i] = 0
        k = min(valid_size, k_query)
        for i in prange(out_size):  # pylint: disable=not-an-iterable
            neigh_indices[i] = i
        tree.query_radius_bottom_up_prealloc(
            coords,
            r2,
            start_nodes,
            dists[:valid_size, :k],
            neigh_indices[:valid_size, :k],
            row_lengths[:valid_size],
        )
        out_size = valid_size
    else:
        # we have to do the sample
        consumed = np.zeros((out_size,), dtype=np.uint8)
        min_dists = np.full((valid_size,), np.inf, dtype=np.float32)
        _ = tree.rejection_ifp_sample_query_prealloc(
            r2,
            r2,
            start_nodes,
            sample_indices,
            dists,
            neigh_indices,
            row_lengths,
            consumed,
            min_dists,
        )
        # sample_result, query_result = tree.ifp_sample_query(
        #     r2, start_nodes, out_size, k_query)

    # sample k_return neighbors and take sqrt
    for i in prange(out_size):  # pylint: disable=not-an-iterable
        rl = row_lengths[i]
        dis = dists[i]
        if rl > k_return:
            indices = np.random.choice(rl, k_return, replace=False)

            # the following is just
            # dists[i, :k_return] = np.sqrt(dists[i, indices])
            # neigh_indices[i, :k_return] = dists[i, :k_return]

            temp_dis = dis[indices]
            ind = neigh_indices[i]
            temp_ind = ind[indices]
            for k in prange(k_return):  # pylint: disable=not-an-iterable
                dis[k] = np.sqrt(temp_dis[k])
                ind[k] = temp_ind[k]

            row_lengths[i] = k_return
        else:
            for k in prange(rl):  # pylint: disable=not-an-iterable
                dis[k] = np.sqrt(dis[k])


print("done")
