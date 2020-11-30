# Numba Neighbors

Approximate port of [scikit-learn neighbors](https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/neighbors) using [Numba](http://numba.pydata.org/).

## Installation

If you want to install/modify (recommented at this point):

```bash
git clone https://github.com/jackd/numba-neighbors.git
pip install -e numba-neighbors
```

Quick-start:

```bash
pip install git+git://github.com/jackd/ifp-sample.git
```

You may see performance benefits from `fastmath` by installing Intel's short vector math library (SVML).

```bash
conda install -c numba icc_rt
```

## Debugging

Debugging is often simpler without `jit`ting. To disable `numba`,

```bash
export NUMBA_DISABLE_JIT=1
```

and re-enable with

```bash
export NUMBA_DISABLE_JIT=0
```

Be wary of using `os.environ["NUMBA_DISABLE_JIT"] = "1"` from python code - this must be set above imports.

## Differences compared to Scikit-learn

1. All operations are done using reduced distances. E.g. provided `KDTree` implementations use squared distances rather than actual distances both for inputs and outputs.
2. `query_radius`-like functions must specify a maximum number of neighbors. Over-estimating this is fairly cheap - it just means we allocate more data than necessary - but if the limit is reached the first `max_count` neighbors that are found are returned. These aren't necessarily the closest `max_count` neighbors.
3. Query outputs aren't sorted, though can be using `binary_tree.simultaneous_sort_partial`.
4. Use of Interl's short vector math library (SVML) if instaled. This makes computation faster, but may result in very small errors.

## TODO

- `n_nodes` is inconsistent with scikit-learn implementation... - scikit bug?
- Port `NeighborsHeap` from scikit learn.
- `query` implementations.
