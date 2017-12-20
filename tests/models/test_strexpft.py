#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy import constants
from lmfit.lineshapes import gaussian, lorentzian

from qef.models.strexpft import StretchedExponentialFTModel

planck_constant = constants.Planck / constants.e * 1E15  # meV*psec

# Test that FT{Gaussian} = Gaussian, and FT{exponential} = Lorentzian
x = np.arange(-0.1, 0.5, 0.0004)  # energy domain, in meV
# items are (tau, beta, intensities). Assumed that tau unit is picoseconds
trios = [(20.0, 2.0,
          gaussian(x, amplitude=1.0, center=0.0,
                   sigma=planck_constant/(np.sqrt(2.0)*20.0*np.pi))),
         (100.0, 1.0,
          # sigma below is actually the HWHM
          lorentzian(x, amplitude=1.0, center=0.0,
                     sigma=planck_constant / (2 * np.pi * 100.0)))
         ]


@pytest.mark.parametrize('tau, beta, y', trios)
def test_lineshapes(tau, beta, y):
    model = StretchedExponentialFTModel()
    params = model.guess(y, x=None)
    # Stray away from optimal parameters
    [params[name].set(value=val) for name, val in
     dict(amplitude=3.0, center=0.0002, tau=50.0, beta=1.5).items()]
    r = model.fit(y, params, x=x)
    all_params = 'tau beta amplitude center'.split()
    assert_allclose([tau, beta, 1.0, 0.0],
                    [r.params[p].value for p in all_params],
                    rtol=0.1, atol=0.000001)


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
