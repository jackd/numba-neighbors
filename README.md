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

## Benchmarks

Requires `scikit-learn >= 0.24.2`

```bash
python benchmarks/kd_tree/query_radius.py
```

- sklearn: `sklearn.neighbors._kd_tree.KDTree` implementation
- numba:   `numba_neighbors.kd_tree.KDTree` implementation
- numba3:  `numba_neighbors.kd_tree.KDTree3` implementation
- bu:      backup implementations (i.e. takes advantage of known leaf indices of query points)
- pre:     pre-allocated memory

```txt
Burning in sklearn...
Benchmarking sklearn...
Burning in numba_pre...
Benchmarking numba_pre...
Burning in numba...
Benchmarking numba...
Burning in numba_bu...
Benchmarking numba_bu...
Burning in numba_bu_pre...
Benchmarking numba_bu_pre...
Burning in numba3...
Benchmarking numba3...
Burning in numba3_bu...
Benchmarking numba3_bu...
Burning in numba3_bu_pre...
Benchmarking numba3_bu_pre...
numba3_bu           : 0.0001566232 (1.00)
numba3_bu_pre       : 0.0001664740 (1.06)
numba3              : 0.0001693816 (1.08)
numba_bu            : 0.0001695879 (1.08)
numba_bu_pre        : 0.0001740832 (1.11)
numba_pre           : 0.0001744671 (1.11)
numba               : 0.0001828313 (1.17)
sklearn             : 0.0008870633 (5.66)
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

Note: this package has served it's purpose for what I wrote it for. I'm unlikely to get to these unless there's serious interest and I get the spare time.

- better benchmarks with [google-benchmark](https://github.com/google/benchmark)
- `n_nodes` is inconsistent with scikit-learn implementation... - scikit bug?
- Port `NeighborsHeap` from scikit learn.
- `query` implementations.
