# TODO

- `radius` bug: `kdtree._update_nodes` does not compute radius values at all when using `numba.prange` and not using the computed values inside the `jit`ted function, maybe because `radius` is never used, so optimizer is pruning it?
- `n_nodes` is inconsistent with scikit-learn implementation... - scikit bug?
- Port `NeighborsHeap` from scikit learn.
