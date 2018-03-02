from __future__ import (absolute_import, division, print_function)

from distutils.version import LooseVersion as version
import numpy as np
from scipy import constants
import lmfit
from lmfit.models import (LorentzianModel, index_of)

planck_constant = constants.Planck / constants.e * 1E15  # meV*psec
hbar = planck_constant / np.pi


class TeixeiraWaterModel(LorentzianModel):
    r"""This fitting function models the dynamic structure factor
    for a particle undergoing jump diffusion.

    J. Teixeira, M.-C. Bellissent-Funel, S. H. Chen, and A. J. Dianoux. `Phys. Rev. A, 31:1913–1917 <http://dx.doi.org/10.1103/PhysRevA.31.1913>`__

    .. math::
        S(Q,E) = \frac{A}{\pi} \cdot \frac{1}{\pi} \frac{\Gamma}{\Gamma^2+(E-E_0)^2}
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
        - diffusion coefficient ``diff`` :math:`D`

    Attributes:
        - Momentum transfer ``q``
    """

    def hwhm_expr(self):
        """Return constraint expression for hwhm"""
        dq2 = '{prefix:s}diff * {q2}'.format(prefix=self.prefix,
                                           q2=self.q * self.q)
        fmt = '{hbar} * dq2/(1 + {prefix:s}tau * dq2})'
        return fmt.format(hbar=hbar, prefix=self.prefix)

    def __init__(self, independent_vars=['x'], prefix='',
                 missing=None, name=None, q=0.0, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing,
                       'independent_vars': independent_vars})
        super(TeixeiraWaterModel, self).__init__(**kwargs)
        self.q = q
        self.set_param_hint('tau', min=0.0)
        self.set_param_hint('diff', min=0.0)
        self.set_param_hint('sigma', expr=self.hwhm_expr)

    if version(lmfit.__version__) > version('0.9.5'):
        __init__.__doc__ = lmfit.models.COMMON_INIT_DOC

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
        diff = 1.0
        if x is not None:
            # Use guess method from the Lorentzian model
            p = super(TeixeiraWaterModel, self).guess(self, y, x)
            amplitude = p['amplitude']
            center = p['center']
            # Assume diff*q*q and tau^(-1) same value
            tau = hbar / (2 * p['sigma'])
            diff = 1.0 / (self.q * self.q * tau)
        return self.make_params(amplitude=amplitude,
                                center=center,
                                tau=tau,
                                diff=diff)
