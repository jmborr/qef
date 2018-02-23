from __future__ import (absolute_import, division, print_function)

import numpy as np
from scipy import constants
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
        - Momentum transfer ``Q``
    """

    def hwhm_expr(self):
        """Return constraint expression for hwhm"""
        fmt = '{hbar}*{prefix:s}diff*{q2}'
        return fmt.format(hbar=hbar, prefix=self.prefix,
                          q2=self.Q * self.Q)

    def __init__(self, independent_vars=['x'], prefix='',
                 missing=None, name=None, Q=0.0, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing,
                       'independent_vars': independent_vars})
        super(TeixeiraWaterModel, self).__init__(**kwargs)
        self.Q = Q
        self.set_param_hint('tau', min=0.0)
        self.set_param_hint('diff', min=0.0)
        self.set_param_hint('sigma', expr=self.hwhm_expr)
