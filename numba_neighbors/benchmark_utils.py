from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from timeit import timeit
import numpy as np


def run_benchmarks(burn_iters, num_iters, *names_and_fns):
    if len(names_and_fns) == 0:
        names_and_fns = base_manager.names_and_fns
    times = []
    names, _ = zip(*names_and_fns)
    for name, fn in names_and_fns:
        print('Burning in {}...'.format(name))
        for _ in range(burn_iters):
            fn()
        print('Benchmarking {}...'.format(name))
        times.append(timeit(fn, number=num_iters) / num_iters)

    indices = np.argsort(times)
    t0 = times[indices[0]]
    for i in indices:
        print('{:20}: {:.10f} ({:.2f})'.format(names[i], times[i],
                                               times[i] / t0))


class BenchmarkManager(object):

    def __init__(self):
        self.names_and_fns = []

    def benchmark(self, name=None):

        def f(decorated):
            actual_name = decorated.__name__ if name is None else name
            self.names_and_fns.append((actual_name, decorated))
            return decorated

        return f

    def run_benchmarks(self, burn_iters, num_iters):
        run_benchmarks(burn_iters, num_iters, *self.names_and_fns)


base_manager = BenchmarkManager()

benchmark = base_manager.benchmark
