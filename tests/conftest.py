from __future__ import print_function, absolute_import

import pytest
import numpy as np
from lmfit.lineshapes import lorentzian


@pytest.fixture(scope='session')
def ltz():
    r"""Sample data is a Lorentzian"""
    x = np.arange(-0.1, 0.5, 0.0004)  # energy domain, in meV
    params = dict(amplitude=1.0, center=0.0, sigma=0.025)
    y = lorentzian(x, **params)
    return dict(x=x, y=y, p=params)
