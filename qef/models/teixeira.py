from __future__ import (absolute_import, division, print_function)

import inspect
from distutils.version import LooseVersion as version
from functools import partial, update_wrapper
import numpy as np
import lmfit
from lmfit.models import (Model, LorentzianModel)
from lmfit.lineshapes import lorentzian

from qef.constants import hbar
from qef.models.utils import MIN_POS_DBL, prefix_params


def teixeira_water(x, amplitude=1.0, center=1.0, tau=1.0, dcf=1.0, q=1.0):
    dq2 = dcf * q * q
    hwhm = hbar * dq2 / (1 + tau * dq2)
    return lorentzian(x, amplitude=amplitude, center=center, sigma=hwhm)


class TeixeiraWaterModel(Model):
    r"""This fitting function models the dynamic structure factor
    for a particle undergoing jump diffusion.

    J. Teixeira, M.-C. Bellissent-Funel, S. H. Chen, and A. J. Dianoux. `Phys. Rev. A, 31:1913-1917 <http://dx.doi.org/10.1103/PhysRevA.31.1913>`__

    .. math::
        S(Q,E) = \frac{A}{\pi} \cdot \frac{\Gamma}{\Gamma^2+(E-E_0)^2}
    .. math::
        \Gamma = \frac{\hbar\cdot D\cdot Q^2}{1+D\cdot Q^2\cdot \tau}

    :math:`\Gamma` is the HWHM of the lorentzian curve.

    At 298K and 1atm, water has :math:`D=2.30 10^{-5} cm^2/s` and
    :math:`\tau=1.25 ps`.

    A jump length :math:`l` can be associated: :math:`l^2=2N\cdot D\cdot \tau`,
    where :math:`N` is the dimensionality of the diffusion problem
    (:math:`N=3` for diffusion in a volume).

    Fitting parameters:
        - integrated intensity ``amplitude`` :math:`A`
        - position of the peak ``center`` :math:`E_0`
        - residence time ``center`` :math:`\tau`
        - diffusion coefficient ``dcf`` :math:`D`

    Attributes:
        - Momentum transfer ``q``
    """  # noqa: E501

    def __init__(self, independent_vars=['x'], q=0.0, prefix='', missing=None,
                 name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'name': name,
                       'independent_vars': independent_vars})
        self.q = q
        def txr(x, amplitude=1.0, center=1.0, tau=1.0, dcf=1.0):
            r"""Teixeira intensities with a particular Q-value

            Not implemented as partial(teixeira_water, q=q) because
            inspect.getargspec does not work with partial objects"""
            return teixeira_water(x, amplitude=amplitude, center=center,
                                  tau=tau, dcf=dcf, q=q)
        super(TeixeiraWaterModel, self).__init__(txr, **kwargs)
        [self.set_param_hint(name, min=MIN_POS_DBL) for name in
         ('amplitude', 'tau', 'dcf')]
        self.set_param_hint('fwhm', expr=self.fwhm_expr)
        self.set_param_hint('height', expr=self.height_expr)

    if version(lmfit.__version__) > version('0.9.5'):
        __init__.__doc__ = lmfit.models.COMMON_INIT_DOC

    @property
    @prefix_params
    def fwhm_expr(self):
        """Constraint expression for FWHM"""
        dq2 = 'dcf*{q2}'.format(q2=self.q * self.q)
        fmt = '2*{hbar}*{dq2}/(1+tau*{dq2})'
        return fmt.format(hbar=hbar, dq2=dq2)

    @property
    @prefix_params
    def height_expr(self):
        """Constraint expression for maximum peak height."""
        fmt = "2.0/{pi}*amplitude/fwhm"
        return fmt.format(pi=np.pi, prefix=self.prefix)

    def guess(self, y, x=None, **kwargs):
        r"""Guess starting values for the parameters of a model.

        Parameters
        ----------
        y : :class:`~numpy:numpy.ndarray`
            Intensities
        x : :class:`~numpy:numpy.ndarray`
            energy values
        kwargs : dict
            additional optional arguments, passed to model function.

        Returns
        -------
        :class:`~lmfit.parameter.Parameters`
            parameters with guessed values
        """
        amplitude = 1.0
        center = 0.0
        tau = 1.0
        dcf = 1.0
        if x is not None:
            # Use guess method from the Lorentzian model
            p = LorentzianModel().guess(y, x)
            center = p['center']
            # Assume diff*q*q and tau^(-1) same value
            tau = hbar / (2 * p['sigma'])
            dcf = 1.0 / (self.q * self.q * tau)
        return self.make_params(amplitude=amplitude,
                                center=center,
                                tau=tau,
                                dcf=dcf)
