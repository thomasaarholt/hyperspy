from hyperspy._components.expression import Expression
from hyperspy.misc.utils import ordinal
import string
import numpy as np

class Polynomial(Expression):

    """n-order polynomial component.

    Polynomial component defined by the coefficients parameters which is an
    array of len the order of the polynomial.

    For example, the [1,2,3] coefficients define the following 3rd order
    polynomial: f(x) = 1x² + 2x + 3

    Attributes
    ----------

    coeffcients : array

    """

    def __init__(self, order=2):
        letters = string.ascii_lowercase
        expr = "+".join(["{}*x**{}".format(letter, power) for letter, power in zip(letters, range(order, -1, -1))])
        name = "{} order Polynomial".format(ordinal(order))
        Expression.__init__(self, expr, name=name)
    
    def get_polynomial_order(self):
        return len(self.parameters) - 1

    def estimate_parameters(self, signal, x1, x2, only_current=False):
        """Estimate the parameters by the two area method

        Parameters
        ----------
        signal : Signal1D instance
        x1 : float
            Defines the left limit of the spectral range to use for the
            estimation.
        x2 : float
            Defines the right limit of the spectral range to use for the
            estimation.

        only_current : bool
            If False estimates the parameters for the full dataset.

        Returns
        -------
        bool

        """
        super(Polynomial, self)._estimate_parameters(signal)
        axis = signal.axes_manager.signal_axes[0]
        binned = signal.metadata.Signal.binned
        i1, i2 = axis.value_range_to_indices(x1, x2)
        if only_current is True:
            estimation = np.polyfit(axis.axis[i1:i2],
                                    signal()[i1:i2],
                                    self.get_polynomial_order())
            if binned is True:
                for para, estim in zip(self.parameters, estimation):
                    para.value = estim / axis.scale
            else:
                for para, estim in zip(self.parameters, estimation):
                    para.value = estim
            return True
        else:
            if self.a.map is None:
                self._create_arrays()
            nav_shape = signal.axes_manager._navigation_shape_in_array
            with signal.unfolded():
                data = signal.data
                # For polyfit the spectrum goes in the first axis
                if axis.index_in_array > 0:
                    data = data.T             # Unfolded, so simply transpose
                fit = np.polyfit(axis.axis[i1:i2], data[i1:i2, ...],
                                   self.get_polynomial_order())
                if axis.index_in_array > 0:
                    fit = fit.T       # Transpose back if needed
                # Shape needed to fit coefficients.map:

                cmap_shape = nav_shape + (self.get_polynomial_order() + 1, )
                fit = fit.reshape(cmap_shape)

                if binned is True:
                    for para, i in zip(self.parameters, range(fit.shape[-1])):
                        para.map['values'][:] = fit[...,i] / axis.scale
                        para.map['is_set'][:] = True
                else:
                    for para, i in zip(self.parameters, range(fit.shape[-1])):
                        para.map['values'][:] = fit[...,i]
                        para.map['is_set'][:] = True
            self.fetch_stored_values()
            return True
