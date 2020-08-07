# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from subprocess import call
from sys import executable

class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    def setup(self):
        pass

    def execute(self, command):
        call((executable, '-c', command))

    def time_hyperspy(self):
        self.execute('import hyperspy.api')

    def time_hyperspy_api_nogui(self):
        self.execute('import hyperspy.api_nogui')
    
    def time_signal(self):
        self.execute('from hyperspy._signals.signal1d import Signal1D')

class MemSuite:
    def mem_list(self):
        return [0] * 256
