#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pytest
import numpy as np
from numpy.testing import assert_allclose

from lmfit.models import LorentzianModel
from qef.models.deltadirac import DeltaDiracModel
from qef.operators.convolve import Convolve


def test_guess():
    x = np.linspace(0, np.pi, 100)  # Energies in meV
    dx = (x[-1] - x[0]) / (len(x) - 1)  # x-spacing
    amplitude = 42.0
    offset = np.pi/6.0
    y = amplitude * np.sin(x + offset)
    p = DeltaDiracModel().guess(y, x=x)
    assert_allclose([amplitude / dx, np.pi/2 - offset],
                    [p['amplitude'], p['center']],
                    rtol=1e-3, atol=1e-5)


def test_convolution():
    amplitude = 42.0
    sigma = 0.042
    center = 0.0003
    c1 = LorentzianModel(prefix='c1_')
    p = c1.make_params(amplitude=amplitude, center=center, sigma=sigma)
    c2 = DeltaDiracModel(prefix='c2_')
    p.update(c2.make_params(amplitude=1.0, center=0.0))
    e = 0.0004 * np.arange(-250, 1500)  # energies in meV
    y1 = Convolve(c1, c2).eval(params=p, x=e)  # should be the lorentzian
    y2 = Convolve(c2, c1).eval(params=p, x=e)  # should be the lorentzian
    m = LorentzianModel()
    all_params = 'amplitude sigma center'.split()
    for y in (y1, y2):
        params = m.guess(y, x=e)
        # Set initial parameters far from optimal solution
        params['amplitude'].set(value=amplitude * 10)
        params['sigma'].set(value=sigma * 4)
        params['center'].set(value=center * 7)
        r = m.fit(y, params, x=e)
        assert_allclose([amplitude, sigma, center],
                        [r.params[p].value for p in all_params],
                        rtol=0.01, atol=0.00001)


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
