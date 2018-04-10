from __future__ import print_function, absolute_import

import os
from os.path import join as pjn
import sys
import pytest
import numpy as np
from lmfit.lineshapes import lorentzian

# Resolve the path to the "external data"
this_module_path = sys.modules[__name__].__file__
data_dir = pjn(os.path.dirname(this_module_path), 'data')


@pytest.fixture(scope='session')
def ltz():
    r"""Sample data is a Lorentzian"""
    x = np.arange(-0.1, 0.5, 0.0004)  # energy domain, in meV
    params = dict(amplitude=1.0, center=0.0, sigma=0.025)
    y = lorentzian(x, **params)
    return dict(x=x, y=y, p=params)


@pytest.fixture(scope='session')
def io_fix():
    return dict(irs_res_f=pjn(data_dir, 'io', 'irs26173_graphite_res.nxs'),
                irs_red_f=pjn(data_dir, 'io', 'irs26176_graphite002_red.nxs'),
                q_values=(0.525312757876, 0.7291668809127, 0.9233951329944,
                          1.105593679447, 1.273206832528, 1.42416584459,
                          1.556455009584, 1.668282739099, 1.758225254224,
                          1.825094271503))
