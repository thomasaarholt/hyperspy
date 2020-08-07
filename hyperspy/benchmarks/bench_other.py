# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

import numpy as np
from numpy.random import Generator, PCG64

from hyperspy.axes import DataAxis, AxesManager
from hyperspy._components.gaussian import Gaussian
from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D


class BenchAxesManager:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    def setup(self): 
        self.ax1 = AxesManager(
            [DataAxis(100*100, navigate=True).get_axis_dictionary(), 
            ])
        self.ax2 = AxesManager(
            [DataAxis(100, navigate=True).get_axis_dictionary(), 
            DataAxis(100, navigate=True).get_axis_dictionary()
            ])

    def time_axes1(self):
        "1D Axes of length 10000"
        for i in self.ax1:
            pass

    def time_axes2(self):
        "2D Axes of length 100x100"
        for i in self.ax2:
            pass

class BenchMultiFit:

    def setup(self): 
        rg = Generator(PCG64(1))
        G = Gaussian(centre=50, sigma=10)
        x = np.arange(100)
        data = G.function(x)
        nav = Signal2D(rg.random((600,))).T
        s = Signal1D(data) * nav
        m = s.create_model()
        m.append(G)
        self.m = m

    def time_multifit(self):
        "Fit a gaussian in 600 positions using the default fitter"
        self.m.multifit()

# class MemSuite:
#     def mem_list(self):
#         return [0] * 256