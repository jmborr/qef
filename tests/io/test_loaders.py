#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import pytest
import os
from numpy.testing import assert_almost_equal
from qef.io import loaders


def test_load_nexus(io_fix):
    data = loaders.load_nexus(io_fix['irs_red_f'])
    assert_almost_equal(data['q'][5], 6.0000, decimal=4)
    assert_almost_equal(data['x'][5], -0.5456, decimal=4)
    assert_almost_equal(data['y'][5][5], 0.0432, decimal=4)
    assert_almost_equal(data['e'][5][5], 0.0070, decimal=4)
    # Intensities for single histogram data should still load as a
    # two-dimensional array
    data = loaders.load_nexus(io_fix['irs_res_f'])
    assert data['y'].shape == (1, 200)


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
