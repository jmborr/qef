#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import pytest
import os
import numpy as np
from numpy.testing import assert_almost_equal

from lmfit.models import LinearModel, LorentzianModel, ConstantModel
import lmfit
from lmfit.model import Model

from qef.constants import hbar  # units of meV x ps  or ueV x ns
from qef.io.loaders import load_nexus
from qef.models.deltadirac import DeltaDiracModel
from qef.models.resolution import TabulatedResolutionModel
from qef.operators.convolve import Convolve


def test_water(io_fix):
    # Load data
    res = load_nexus(io_fix['irs_res_f'])
    dat = load_nexus(io_fix['irs_red_f'])
    q_vals = io_fix['q_values']

    # Define the fitting range
    e_min = -0.4
    e_max = 0.4
    # Find indexes of dat['x'] with values in (e_min, e_max)
    mask = np.intersect1d(np.where(dat['x'] > e_min),
                          np.where(dat['x'] < e_max))
    # Drop data outside the fitting range
    fr = dict()  # fitting range. Use in place of 'dat'
    fr['x'] = dat['x'][mask]
    fr['y'] = np.asarray([y[mask] for y in dat['y']])
    fr['e'] = np.asarray([e[mask] for e in dat['e']])

    # Create the model
    def generate_model_and_params(spectrum_index=None):
        r"""Produce an LMFIT model and related set of fitting parameters"""

        sp = '' if spectrum_index is None else '{}_'.format(
            spectrum_index)  # prefix if spectrum_index passed

        # Model components
        intensity = ConstantModel(prefix='I_' + sp)  # I_amplitude
        elastic = DeltaDiracModel(prefix='e_' + sp)  # e_amplitude, e_center
        # l_amplitude, l_center, l_sigma (also l_fwhm, l_height)
        inelastic = LorentzianModel(prefix='l_' + sp)
        # r_amplitude, r_center (both fixed)
        resolution = TabulatedResolutionModel(res['x'], res['y'],
                                              prefix='r_' + sp)
        background = LinearModel(prefix='b_' + sp)  # b_slope, b_intercept

        # Putting it all together
        model = intensity * Convolve(resolution,
                                     elastic + inelastic) + background
        parameters = model.make_params()  # model params are a separate entity

        # Ties and constraints
        parameters['e_' + sp + 'amplitude'].set(min=0.0, max=1.0)
        parameters['l_' + sp + 'center'].set(
            expr='e_' + sp + 'center')  # centers tied
        parameters['l_' + sp + 'amplitude'].set(
            expr='1 - e_' + sp + 'amplitude')

        # Some initial sensible values
        init_vals = {'I_' + sp + 'c': 1.0, 'e_' + sp + 'amplitude': 0.5,
                     'l_' + sp + 'sigma': 0.01,
                     'b_' + sp + 'slope': 0, 'b_' + sp + 'intercept': 0}
        for p, v in init_vals.items():
            parameters[p].set(value=v)

        return model, parameters
    # Call the function
    model, params = generate_model_and_params()

    # Initial guess for first spectrum. Only set free parameters
    for name, value in dict(I_c=4.0, e_center=0, e_amplitude=0.1,
                            l_sigma=0.03, b_slope=0, b_intercept=0).items():
        params[name].set(value=value)
    # Carry out the fit
    fit = model.fit(fr['y'][0], x=fr['x'], params=params,
                    weights=1.0 / fr['e'][0])
    assert_almost_equal(fit.redchi, 1.72, decimal=2)

    # Carry out sequential fit
    n_spectra = len(fr['y'])
    fits = [None, ] * n_spectra  # store fits for all the tried spectra
    fits[0] = fit  # store previous fit
    for i in range(1, n_spectra):
        y_exp = fr['y'][i]
        e_exp = fr['e'][i]
        fit = model.fit(y_exp, x=fr['x'], params=params, weights=1.0 / e_exp)
        fits[i] = fit  # store fit results
    assert_almost_equal([f.redchi for f in fits],
                        [1.72, 1.15, 0.81, 0.73, 0.73, 0.75, 0.81, 0.86, 0.75,
                         0.91],
                        decimal=2)

    # Fit HWHM(Q^2) with Teixeira model
    hwhms = 0.5 * np.asarray([fit.params['l_fwhm'].value for fit in fits])

    def teixeira(q2s, difcoef, tau):
        dq2 = difcoef * q2s
        return hbar * dq2 / (1 + dq2 * tau)
    teixeira_model = Model(teixeira)  # create LMFIT Model instance
    teixeira_model.set_param_hint('difcoef', min=0)
    teixeira_model.set_param_hint('tau', min=0)
    # Carry out the fit from an initial guess
    teixeira_params = teixeira_model.make_params(difcoef=1.0, tau=1.0)
    teixeira_fit = teixeira_model.fit(hwhms, q2s=np.square(q_vals),
                                      params=teixeira_params)
    assert_almost_equal([teixeira_fit.best_values['difcoef'],
                         teixeira_fit.best_values['tau']],
                        [0.16, 1.11], decimal=2)

    # Model for Simultaneous Fit of All Spectra with Teixeira Water Model
    #
    # create one model for each spectrum, but collect all parameters under
    # a single instance of the Parameters class.
    l_model = list()
    g_params = lmfit.Parameters()
    for i in range(n_spectra):
        # model and parameters for one of the spectra
        m, ps = generate_model_and_params(spectrum_index=i)
        l_model.append(m)
        [g_params.add(p) for p in ps.values()]

    # Initialize parameter set with optimized parameters from sequential fit
    for i in range(n_spectra):
        optimized_params = fits[i].params  # these are I_c, e_amplitude,...
        for name in optimized_params:
            # for instance, 'e_amplitude' splitted into 'e', and 'amplitude'
            prefix, base = name.split('_')
            # i_name is 'e_3_amplitude' for i=3
            i_name = prefix + '_{}_'.format(i) + base
            g_params[i_name].set(value=optimized_params[name].value)

    # Introduce global parameters difcoef and tau.
    # Use previous optimized values as initial guess
    o_p = teixeira_fit.params
    g_params.add('difcoef', value=o_p['difcoef'].value, min=0)
    g_params.add('tau', value=o_p['tau'].value, min=0)

    # Tie each lorentzian l_i_sigma to the teixeira expression
    for i in range(n_spectra):
        q2 = q_vals[i] * q_vals[i]
        fmt = '{hbar}*difcoef*{q2}/(1+difcoef*{q2}*tau)'
        teixeira_expression = fmt.format(hbar=hbar, q2=q2)
        g_params['l_{}_sigma'.format(i)].set(expr=teixeira_expression)

    # Carry out the Simultaneous Fit
    def residuals(params):
        l_residuals = list()
        for i in range(n_spectra):
            x = fr['x']  # fitting range of energies
            y = fr['y'][i]  # associated experimental intensities
            e = fr['e'][i]  # associated experimental errors
            model_evaluation = l_model[i].eval(x=x, params=params)
            l_residuals.append((model_evaluation - y) / e)
        return np.concatenate(l_residuals)
    # Minimizer object using the parameter set for all models and the
    # function to calculate all the residuals.
    minimizer = lmfit.Minimizer(residuals, g_params)
    g_fit = minimizer.minimize()
    assert_almost_equal(g_fit.redchi, 0.93, decimal=2)


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
