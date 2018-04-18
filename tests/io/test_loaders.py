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


def test_load_dave(io_fix):
    data = loaders.load_dave(io_fix['dave'], to_meV=False)
    #  assert Q values
    assert (data['q'][0], data['q'][-1]) == (0.3, 1.9)
    # assert energies
    assert (data['x'][0], data['x'][-1]) == (-119.8, 119.8)
    # assert fist spectrum
    assert (data['y'][0][0], data['y'][0][-1]) == (0.00589672, 0.00789678)
    assert (data['e'][0][0], data['e'][0][-1]) == (0.000488621, 0.000631035)
    # assert last spectrum
    assert (data['y'][-1][0], data['y'][-1][-1]) == (0.0128807, 0.0154997)
    assert (data['e'][-1][0], data['e'][-1][-1]) == (0.000419397, 0.000421017)


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
