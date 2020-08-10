from hyperspy._signals.signal1d import Signal1D
from numpy import zeros

def axes_iteration(s):
    for i in s.axes_manager:
        pass
    
def test_bench_axes_1000(benchmark):
    "Test iterating through 1000 indices"
    s = Signal1D(zeros((10, 10, 10, 1)))
    benchmark.pedantic(axes_iteration, args = (s,), rounds=5, iterations=5)
    
    
 