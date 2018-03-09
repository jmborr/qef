#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import os
from numpy.testing import assert_almost_equal, assert_allclose
import pytest

from qef.models.teixeira import TeixeiraWaterModel
from qef.constants import hbar


def test_init():
    tx = TeixeiraWaterModel(prefix='tx_', q=0.3)
    assert 'tx_dcf*0.09/(1+tx_tau*tx_dcf*0.09)'\
           in tx.param_hints['fwhm']['expr']


def test_guess(ltz):
    tx = TeixeiraWaterModel(q=0.3)
    p = tx.guess(ltz['y'], x=ltz['x'])
    assert_almost_equal(p['fwhm'], 2 * ltz['p']['sigma'], decimal=2)
    assert_almost_equal(hbar / p['tau'], 2 * ltz['p']['sigma'], decimal=2)


def test_fit(ltz):
    tx = TeixeiraWaterModel(q=0.3)
    p = tx.guess(ltz['y'], x=ltz['x'])
    p['tau'].value *= 2.0  # get away fro
    fr = tx.fit(ltz['y'], p, x=ltz['x'])
    assert_almost_equal(fr.params['fwhm'], 2 * ltz['p']['sigma'], decimal=6)


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])




