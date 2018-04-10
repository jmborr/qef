from __future__ import (absolute_import, division, print_function)

import numpy as np
from scipy import constants

planck_constant = constants.Planck / constants.e * 1E15  # meV*psec
hbar = planck_constant / (2 * np.pi)  # meV*psec
