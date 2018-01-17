from __future__ import (absolute_import, division, print_function)

import os
import pytest
import numpy as np
from numpy.testing import assert_allclose
from lmfit.lineshapes import lorentzian
from qef.models.tabulatedmodel import TabulatedModel


def test_tabulation():
    x_sim = np.arange(-1.0, 1.0, 0.0003)  # energy domain, in meV
    y_sim = lorentzian(x_sim , amplitude=1, center=0, sigma=0.042)
    intensity = 42.0
    peak_center = 0.0002
    x_exp = np.arange(-0.1, 0.5, 0.0004)
    y_exp = lorentzian(x_exp - 0, amplitude=intensity, center=peak_center, sigma=0.042)

    model = TabulatedModel(x_sim,y_sim)
    params = model.guess(x_exp, y_exp)
    fit = model.fit(y_exp, params, x=x_exp, fit_kws={'nan_policy': 'omit'})

    assert abs(fit.best_values['amplitude'] - intensity) < 0.0001
    assert abs(fit.best_values['center'] - peak_center) < 0.0001

if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])









