#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import os
import numpy as np
from numpy.testing import assert_almost_equal
from lmfit.lineshapes import lorentzian
from scipy import constants
import pytest

from qef.models.teixeira import TeixeiraWaterModel

planck_constant = constants.Planck / constants.e * 1E15  # meV*psec


def test_init():
    tx = TeixeiraWaterModel(prefix='tx_', q=0.3)
    assert 'tx_diff * 0.09/(1 + tx_tau * tx_diff * 0.09)'\
           in tx.param_hints['sigma']['expr']()


def test_guess():
    tx = TeixeiraWaterModel(q=0.3)
    hwhm = 0.025
    x = np.arange(-0.1, 0.5, 0.0004)  # energy domain, in meV
    y = lorentzian(x, amplitude=1.0, center=0.0,
                   sigma=planck_constant / (2 * np.pi * hwhm))
    p = tx.guess(y, x=x)
    assert_almost_equal(p['sigma'], hwhm, decimal=3)



if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])




