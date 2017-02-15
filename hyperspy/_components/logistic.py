# -*- coding: utf-8 -*-
# Copyright 2007-2020 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

from hyperspy._components.expression import Expression


class Logistic(Expression):

    r"""Logistic function (sigmoid or s-shaped curve) component.

    .. math::
    
        f(x) = \frac{a}{1 + b\cdot \mathrm{exp}\left[-c 
            \left((x - x_0\right)\right]}

    ============== =============
    Variable        Parameter 
    ============== =============
    :math:`A`       a 
    :math:`b`       b 
    :math:`c`       c 
    :math:`x_0`     origin 
    ============== =============


    Parameters
    -----------
    a : Float
        The curve's maximum y-value,  
        :math:`\mathrm{lim}_{x\to\infty}\left(y\right) = a`
    b : Float
        Additional parameter: 
        b>1 shifts origin to larger values;
        0<b<1 shifts origin to smaller values;
        b<0 introduces an asymptote
    c : Float
        Logistic growth rate or steepness of the curve
    origin : Float
        Position of the sigmoid's midpoint
    **kwargs
        Extra keyword arguments are passed to the ``Expression`` component.
    """

<<<<<<< HEAD
    def __init__(self, a=1., b=1., c=1., origin=0., module="numexpr", **kwargs):
        super(Logistic, self).__init__(
            expression="a / (1 + b * exp(-c * (x - origin)))",
            name="Logistic",
            a=a,
            b=b,
            c=c,
            origin=origin,
            position="origin",
            module=module,
            autodoc=False,
            **kwargs)

        # Boundaries
        self.isbackground = False
=======
    def __init__(self):
        # Define the parameters
        Component.__init__(self, ('a', 'b', 'c', 'origin'))
        # Define the name of the component
        self.a.grad = self.grad_a
        self.b.grad = self.grad_b
        self.c.grad = self.grad_c
        self.origin.grad = self.grad_origin
        self._position = self.origin

        # Linearity
        self.a._is_linear = True
        
    def function(self, x):
        """
        """
        a = self.a.value
        b = self.b.value
        c = self.c.value
        origin = self.origin.value
        return a / (1 + b * np.exp(-c * (x - origin)))

    def grad_a(self, x):
        """
        Returns d(function)/d(parameter_1)
        """
        b = self.b.value
        c = self.c.value
        origin = self.origin.value

        return 1 / (1 + b * np.exp(-c * (x - origin)))

    def grad_b(self, x):
        """
        Returns d(function)/d(parameter_1)
        """
        a = self.a.value
        b = self.b.value
        c = self.c.value
        origin = self.origin.value

        return -(a * np.exp(-c * (x - origin))) / \
            (b * np.exp(-c * (x - origin)) + 1) ** 2

    def grad_c(self, x):
        """
        Returns d(function)/d(parameter_1)
        """
        a = self.a.value
        b = self.b.value
        c = self.c.value
        origin = self.origin.value

        return -(a * b * (origin - x) * np.exp(-c * (x - origin))) / \
            (b * np.exp(-c * (x - origin)) + 1) ** 2

    def grad_origin(self, x):
        """
        Returns d(function)/d(parameter_1)
        """
        a = self.a.value
        b = self.b.value
        c = self.c.value
        origin = self.origin.value

        return -(a * b * c * np.exp(-c * (x - origin))) / \
            (b * np.exp(-c * (x - origin)) + 1) ** 2
>>>>>>> Linearity attribute to parameters
