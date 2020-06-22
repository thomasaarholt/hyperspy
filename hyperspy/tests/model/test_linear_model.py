from unittest import mock

import numpy as np
import pytest

from hyperspy.misc.eels.eelsdb import eelsdb
from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.signal2d import Signal2D

from hyperspy.components1d import Gaussian, PowerLaw, Expression, Offset, \
Expression, Offset, ScalableFixedPattern, Lorentzian, Arctan, \
Erf, Exponential, HeavisideStep, Logistic, PESCoreLineShape, SEE, \
RC, VolumePlasmonDrude


from hyperspy.components2d import Gaussian2D

from hyperspy.datasets.example_signals import EDS_SEM_Spectrum
from hyperspy.datasets.artificial_data import get_low_loss_eels_signal
from hyperspy.datasets.artificial_data import get_core_loss_eels_signal
from hyperspy.misc.utils import slugify
from hyperspy.decorators import lazifyTestClass

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
        Model is not currently linear as Gaussian sigma and centre parameters
        are free
        """
        assert self.m.nonlinear_parameters

    def test_fit_linear(self):
        self.m[0].sigma.free = False
        self.m[0].centre.free = False
        self.m.fit(fitter="linear")
        np.testing.assert_allclose(self.m[0].A.value, 6132.640632924692, 1)
        np.testing.assert_allclose(self.m[0].centre.value, 0.5)
        np.testing.assert_allclose(self.m[0].sigma.value, 1)


class TestMultiFitLinear:

    def setup_method(self):
        np.random.seed(1)
        x = np.random.random(30)
        shape = np.random.random((2,3,1))
        X = shape*x
        self.s = Signal1D(X)
        self.m = self.s.create_model()

    def test_gaussian(self):
        m = self.m
        L = Gaussian(centre=15.)
        L.centre.free = L.sigma.free = False
        m.append(L)

        m.fit(fitter='linear')
        single = m.as_signal()
        m.assign_current_values_to_all()
        m.multifit(fitter='linear', iterpath='flyback')
        multi = m.as_signal()

        np.testing.assert_almost_equal(
            single.inav[0,0].data, multi.inav[0,0].data)

    def test_offset(self):
        m = self.m
        L = Offset(offset=1.)
        m.append(L)

        m.fit(fitter='linear')
        single = m.as_signal()
        m.assign_current_values_to_all()
        m.multifit(fitter='linear', iterpath='flyback')
        multi = m.as_signal()
        # compare fits from first pixel
        np.testing.assert_almost_equal(
            single.inav[0,0].data, multi.inav[0,0].data)

class TestLinearFitting:
    def setup_method(self, method):
        self.s = EDS_SEM_Spectrum().isig[5.0:15.0]
        self.m = self.s.create_model(auto_background=False)
        self.c = Expression('a*x+b', 'line with offset')
        self.m.append(self.c)

    def test_linear_fitting_with_offset(self):
        m = self.m
        m.fit(fitter='linear')
        linear = m.as_signal()
        np.testing.assert_allclose(m.p0, np.array([933.2343071493418, 47822.98004150301, -5867.611808815612, 56805.518919752234]))

        # Repeat test with offset fixed
        self.c.b.free = False
        m.fit(fitter='linear')
        linear = m.as_signal()
        np.testing.assert_allclose(m.p0, np.array([933.2343071496773, 47822.98004150315, -5867.611808815624]))

    def test_fixed_offset_value(self):
        self.m.fit(fitter='linear')
        c = self.c
        c.b.free = False
        constant = c._compute_constant_term()
        np.testing.assert_array_almost_equal(constant, c.b.value)

class TestFitAlgorithms:
    def setup_method(self, method):
        self.s = EDS_SEM_Spectrum().isig[5.0:15.0]
        self.m = self.s.create_model(auto_background=False)
        self.c = Expression('a*x+b', 'line with offset')
        self.m.append(self.c)

    def test_compare_algorithms(self):
        m = self.m
        m.fit(linear_algorithm='ridge_regression')
        assert m._linear_algorithm == 'ridge_regression'

        ridge_fit = m.as_signal()

        m.fit(linear_algorithm='matrix_inversion')
        assert m._linear_algorithm == 'matrix_inversion'
        matrix_fit = m.as_signal()
        np.testing.assert_array_almost_equal(ridge_fit.data, matrix_fit.data)

class TestLinearEELSFitting:
    def setup_method(self, method):
        self.ll = get_low_loss_eels_signal()
        self.cl = get_core_loss_eels_signal()
        self.cl.add_elements(('Mn',))
        self.m = self.cl.create_model(auto_background=False)
        self.m[0].onset_energy.value = 673.
        self.m_convolved = self.cl.create_model(auto_background=False, ll=self.ll)
        self.m_convolved[0].onset_energy.value = 673.

    def test_convolved_and_std_error(self):
        m = self.m_convolved
        m.fit(fitter='linear')
        linear = m.as_signal()
        std_linear = m.p_std
        m.fit(fitter='leastsq')
        leastsq = m.as_signal()
        std_leastsq = m.p_std
        diff = linear - leastsq
        np.testing.assert_almost_equal(diff.data.sum(), 0.0, decimal=2)
        np.testing.assert_almost_equal(std_linear, std_leastsq, decimal=5)

    def test_nonconvolved(self):
        m = self.m
        m.fit(fitter='linear')
        linear = m.as_signal()
        m.fit(fitter='leastsq')
        leastsq = m.as_signal()
        diff = linear - leastsq
        np.testing.assert_almost_equal(diff.data.sum(), 0.0, decimal=2)

class TestLinearModel2D:
    def setup_method(self, method):
        low, high = -10, 10
        N = 100
        self.x = self.y = np.linspace(low, high, N)
        self.mesh = np.meshgrid(self.x, self.y)

    def test_model2D_one_component(self):
        G1 = Gaussian2D(30, 5.0, 4.0, 0, 0)

        data = G1.function(*self.mesh)
        s = Signal2D(data)
        s.axes_manager[-2].offset = self.x[0]
        s.axes_manager[-1].offset = self.y[0]

        s.axes_manager[-2].scale = self.x[1] - self.x[0]
        s.axes_manager[-1].scale = self.y[1] - self.y[0]

        m = s.create_model()
        m.append(G1)

        G1.set_parameters_not_free()
        G1.A.free = True

        m.fit(fitter='linear')
        diff = (s - m.as_signal(show_progressbar=False))
        np.testing.assert_almost_equal(diff.data.sum(), 0.0)
        np.testing.assert_almost_equal(m.p_std[0], 0.0)

    def test_model2D_linear_many_gaussians(self):
        gausslow, gausshigh = -8, 8
        gauss_step = 8
        X, Y = self.mesh
        z = np.zeros(X.shape)
        g = Gaussian2D()
        for i in np.arange(gausslow, gausshigh+1, gauss_step):
            for j in np.arange(gausslow, gausshigh+1, gauss_step):
                g.centre_x.value = i
                g.centre_y.value = j
                g.A.value = 10
                z += g.function(X, Y)

        s = Signal2D(z)
        s.axes_manager[-2].offset = self.x[0]
        s.axes_manager[-1].offset = self.y[0]

        s.axes_manager[-2].scale = self.x[1] - self.x[0]
        s.axes_manager[-1].scale = self.y[1] - self.y[0]

        m = s.create_model()
        for i in np.arange(gausslow, gausshigh+1, gauss_step):
            for j in np.arange(gausslow, gausshigh+1, gauss_step):
                g = Gaussian2D(centre_x = i, centre_y=j)
                g.set_parameters_not_free()
                g.A.free = True
                m.append(g)

        m.fit(fitter='linear')
        np.testing.assert_array_almost_equal(s.data, m.as_signal().data)

    def test_model2D_polyexpression(self):
        poly = "a*x**2 + b*x - c*y**2 + d*y + e"
        P = Expression(poly, 'poly')
        P.a.value = 6
        P.b.value = 5
        P.c.value = 4
        P.d.value = 3
        P.e.value = 2
        
        data = P.function(*self.mesh)# + G2.function(*mesh)
        s = Signal2D(data)

        m = s.create_model()
        m.append(P)
        m.fit(fitter='linear')
        diff = (s - m.as_signal(show_progressbar=False))
        np.testing.assert_almost_equal(diff.data.sum(), 0.0, decimal=2)
        np.testing.assert_almost_equal(m.p_std, 0.0, decimal=2)

class TestLinearFitTwins:
    def setup_method(self, method):
        from hyperspy._components.gaussian import Gaussian
        from hyperspy._signals.signal1d import Signal1D
        g1 = Gaussian(centre=10)
        g2 = Gaussian(centre=20)
        g3 = Gaussian(centre=30)

        g3.A.twin = g2.A
        g3.A.twin_function_expr = "-0.5*x"
        g2.A.twin = g1.A
        g2.A.twin_function_expr = "-0.5*x"

        g1.A.value = 20
        x = np.linspace(0, 50, 1000)

        y = g1.function(x) + g2.function(x) + g3.function(x)
        s = Signal1D(y)
        s.axes_manager[-1].scale = x[1] - x[0]

        gs = [g1, g2, g3]
        m = s.create_model()
        m.extend(gs)
        self.s, self.m, self.gs = s, m, gs

    def test_without_twins(self):
        for g in self.gs:
            g.sigma.free = False
            g.centre.free = False
            g.A.twin = None

        self.gs[0].A.value = 1
        self.m.fit(fitter='linear')

        np.testing.assert_almost_equal(self.gs[0].A.value, 20)
        np.testing.assert_almost_equal(self.gs[1].A.value, -10)
        np.testing.assert_almost_equal(self.gs[2].A.value, 5)
        np.testing.assert_array_almost_equal((self.s - self.m.as_signal()).data, 0)


    def test_with_twins(self):
        for g in self.gs:
            g.sigma.free = False
            g.centre.free = False

        self.gs[0].A.value = 1
        self.m.fit(fitter='linear')
        
        np.testing.assert_almost_equal(self.gs[0].A.value, 20)
        np.testing.assert_almost_equal(self.gs[1].A.value, -10)
        np.testing.assert_almost_equal(self.gs[2].A.value, 5)
        np.testing.assert_array_almost_equal((self.s - self.m.as_signal()).data, 0)
