#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import pytest
import os
import numpy as np
from lmfit.models import LinearModel, LorentzianModel
import lmfit

from qef.io.loaders import load_nexus
from qef.models.deltadirac import DeltaDiracModel
from qef.models.tabulatedmodel import TabulatedModel
from qef.models.resolution import TabulatedResolutionModel
from qef.models.teixeira import TeixeiraWaterModel
from qef.operators.convolve import Convolve


def test_water(io_fix):
    fwhms = [0.0571426119902978, 0.0999618315234958, 0.15069089187027496,
             0.20499653224019898, 0.25574438060694726, 0.3083128897367793,
             0.3533967368449584, 0.38644048123693686, 0.41727343549036133,
             0.4258484471724864]
    q_vals = [0.52531276, 0.72916688, 0.92339513, 1.10559368, 1.27320683,
              1.42416584, 1.55645501, 1.66828274, 1.75822525, 1.82509427]
    # Create the model
    from qef.constants import hbar  # units of meV x ps  or ueV x ns
    from lmfit.model import Model

    def teixeira(x, diff, tau):
        r"""Calculate FWHM for a given Q, diffusion coefficient, and
        relaxation time"""
        xarr = np.asarray(x)
        dq2 = diff * xarr * xarr
        return 2 * hbar * dq2 / (1 + dq2 * tau)

    teixeira_model = Model(teixeira)  # create LMFIT Model instance
    teixeira_model.set_param_hint('diff',
                                  min=0)  # diffusion coeff must be positive
    teixeira_model.set_param_hint('tau',
                                  min=0)  # relaxation coeff must be positive

    # Carry out the fit
    teixeira_params = teixeira_model.make_params(diff=0.2,
                                                 tau=1.0)  # initial guess
    teixeira_params.add('junk', value=0.5)
    teixeira_params['tau'].set(expr='2*junk')
    me = teixeira_model.eval(x=q_vals, params=teixeira_params)
    teixeira_fit = teixeira_model.fit(fwhms, x=q_vals, params=teixeira_params)

    # Loading data
    #
    # resolution has only one histogram
    res = load_nexus(io_fix['irs_res_f'])
    # data has 10 histograms
    dat = load_nexus(io_fix['irs_red_f'])
    qs = io_fix['q_values']

    elastic = TabulatedModel(res['x'], res['y'], prefix='e_')
    inelastic = LorentzianModel(prefix='q_')
    resolution = TabulatedResolutionModel(res['x'], res['y'], prefix='r_')
    elastic.set_param_hint('e_amplitude', value=0.0)
    inelastic.set_param_hint('q_amplitude', value=0.0)
    model = elastic + Convolve(resolution, inelastic)
    params = model.make_params()
    y = model.eval(x=dat['x'], params=params)

    elastic = TabulatedModel(res['x'], res['y'])
    model = elastic
    params = model.make_params()
    model_evaluation = model.eval(x=dat['x'], params=params)
    fits = list()
    for i, q in enumerate(qs):
        p = '{}_'.format(i)
        elastic = TabulatedModel(res['x'], res['y'], prefix='e_' + p)
        quasi = LorentzianModel(prefix='q_' + p)
        background = LinearModel(prefix='b_' + p)
        background.set_param_hint('slope', value=0.01, vary=True)
        background.set_param_hint('intercept', value=0.02, vary=True)
        model = elastic + quasi + background
        params = model.make_params()
        fit = model.fit(dat['y'][i], x=dat['x'], params=params)
        #, weights = 1.0 / dat['e'][i]
        fits.append(fit)

    # Sequential fit with lorentzian model
    #
    fits = list()
    for i, q in enumerate(qs):
        p = '{}_'.format(i)
        elastic = TabulatedModel(res['x'], res['y'], prefix='e_' + p)
        resolution = TabulatedResolutionModel(res['x'], res['y'], prefix='r_' + p)
        quasi = Convolve(resolution, LorentzianModel(prefix='q_' + p))
        background = LinearModel(prefix='b_' + p)
        model = elastic + quasi + background
        params = model.make_params()
        fit = model.fit(dat['y'][i], weights=1.0/dat['e'][i], x=dat['x'],
                        params=params)
        fits.append(fit)


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
