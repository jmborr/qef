#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lmfit.models import TabulatedModel
from lmfit.lineshapes import lorentzian

# Simulated data
x_sim = np.arange(-0.1, 0.5, 0.0004)  # energy domain, in meV
y_sim = lorentzian(x_sim, amplitude=1.0, center=0.0, sigma=0.042)

# Experimental data is the same Lorentzian with different amplitude and
# shifted
intensity = 42.0
peak_center = 0.0002
y_exp = lorentzian(x_sim - shift, amplitude=intensity,
                   center=peak_center, sigma=0.042)

# To-Do: create interpolator with simulated data and fit against experimental
# data so that amplitude==intensity and center==peak_center

