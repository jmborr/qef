#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import pytest
import os
from lmfit.models import LinearModel
import lmfit

from qef.io.loaders import load_nexus
from qef.models import DeltaDiracModel, TabulatedModel, TeixeiraWaterModel
from qef.operators.convolve import Convolve

def test_water(io_fix):
    # Loading data
    #
    # resolution has only one histogram
    res = load_nexus(io_fix['irs_res_f'])
    # data has 10 histograms
    dat = load_nexus(io_fix['irs_red_f'])
    qs = io_fix['q_values']


    # Sequential fit with teixeira model
    #
    for i, q in enumerate(qs):
        p = '{}_'.format(i)
        elastic = TabulatedModel(res['x'], res['y'], prefix='e_' + p)
        resolution = TabulatedModel(res['x'], res['y'], prefix='r_' + p)
        resolution.set_param_hint('amplitude', vary=False)
        resolution.set_param_hint('centre', vary=False)
        quasi = Convolve(resolution, TeixeiraWaterModel(prefix='q_' + p, q=q))
        background = LinearModel(prefix='b_' + p)
        model = elastic + quasi + background
        params = model.guess(dat['y'][i], x=dat['x'])
        fit = model.fit(dat['y'][i], params=params, x=dat['x'])





if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
