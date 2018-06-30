from unittest import mock

import numpy as np
import pytest

from hyperspy.misc.eels.eelsdb import eelsdb
from hyperspy._signals.signal1d import Signal1D
from hyperspy._components.gaussian import Gaussian
from hyperspy._components.power_law import PowerLaw
from hyperspy._components.expression import Expression

from hyperspy.datasets.example_signals import EDS_SEM_Spectrum
from hyperspy.misc.utils import slugify
from hyperspy.decorators import lazifyTestClass

@lazifyTestClass
class TestModelFitBinned:

    def setup_method(self, method):
        np.random.seed(1)
        s = Signal1D(
            np.random.normal(
                scale=2,
                size=10000)).get_histogram()
        s.metadata.Signal.binned = True
        g = Gaussian()
        m = s.create_model()
        m.append(g)
        g.sigma.value = 1
        g.centre.value = 0.5
        g.A.value = 1e3
        self.m = m

    def test_model_is_not_linear(self):
        """
        Model is not currently linear as Gaussian sigma and centre parameters are free
        """
        assert not self.m._check_all_active_components_are_linear()

    def test_fit_lsq_linear(self):
        self.m[0].sigma.free = False
        self.m[0].centre.free = False
        self.m.fit(fitter="linear")
        np.testing.assert_allclose(self.m[0].A.value, 6132.640632924692, 1)
        np.testing.assert_allclose(self.m[0].centre.value, 0.5)
        np.testing.assert_allclose(self.m[0].sigma.value, 1)

@lazifyTestClass
class TestMultifit:

    def setup_method(self, method):
        s = Signal1D(np.zeros((2, 200)))
        s.axes_manager[-1].offset = 1
        s.data[:] = 2 * s.axes_manager[-1].axis ** (-3)

        m = s.create_model()
        m.append(PowerLaw())
        m[0].A.value = 2
        m[0].r.value = 2
        m.store_current_values()
        m.axes_manager.indices = (1,)
        m[0].r.value = 100
        m[0].A.value = 2
        m.store_current_values()
        m[0].A.free = False
        self.m = m
        m.axes_manager.indices = (0,)
        m[0].A.value = 100

    def test_bounded_lsq_linear(self):
        m = self.m
        m[0].A.free = True
        m[0].r.free = False

        m.signal.data *= 2.
        m[0].A.value = 2.
        m[0].r.value = 3.
        m[0].r.assign_current_value_to_all()
        m.multifit(fitter='linear', bounded=True, show_progressbar=None)
        np.testing.assert_array_almost_equal(self.m[0].A.map['values'],
                                             [4., 4.])

class TestLinearFitting:
    def setup_method(self, method):
        self.s = EDS_SEM_Spectrum().isig[5.0:15.0]
        self.m = self.s.create_model(auto_background=False)
        self.c = Expression('a*x+b', 'Linear')
        self.m.append(self.c)

    def test_linear_fitting_with_offset(self):
        m = self.m
        m.fit('linear')
        linear = m.as_signal()
        np.testing.assert_allclose(m.p0, np.array([   933.234307,  47822.980041,  -5867.611809,  56805.51892 ]))
        
        m.fit('leastsq')
        leastsq = m.as_signal()
        diff = (leastsq - linear)
        assert diff.data.sum() == 0

    def test_free_offset_value(self):
        c = self.c
        c.a.free = False
        assert c._compute_free_offset_parameter_value() == c.b.value

    def test_fixed_offset_value(self):
        c = self.c
        m = self.m
        assert (c._compute_constant_term() - c.b.value*np.ones(m.axis.axis.shape)).sum() == 0

class TestLinearEELSFitting:
    def setup_method(self, method):
        self.ll, self.cl = eelsdb(title = 'Niobium Oxide', author='Wilfred Sigle')[:2]
        self.cl2 = self.cl.remove_background((100.,200.))
        self.m = self.cl2.create_model(auto_background=False)
        self.m_convolved = self.cl2.create_model(auto_background=False, ll=self.ll)

    def test_convolved(self):
        m = self.m_convolved
        m.fit('linear')
        linear = m.as_signal()
        m.fit('leastsq')
        leastsq = m.as_signal()
        diff = linear - leastsq
        np.testing.assert_almost_equal(diff.data.sum(), 0.0, decimal=2)

    def test_nonconvolved(self):
        m = self.m
        m.fit('linear')
        linear = m.as_signal()
        m.fit('leastsq')
        leastsq = m.as_signal()
        diff = linear - leastsq
        np.testing.assert_almost_equal(diff.data.sum(), 0.0, decimal=2)

    def test_chained_twins(self):
        m = self.m
        m[4].parameters[0].twin = m[3].parameters[0]
        m[3].parameters[0].twin = m[2].parameters[0]
        m.fit('linear')
        linear = m.as_signal()
        m.fit('leastsq')
        leastsq = m.as_signal()
        diff = linear - leastsq
        np.testing.assert_almost_equal(diff.data.sum(), 0.0, decimal=2)
