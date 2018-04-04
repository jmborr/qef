from __future__ import (absolute_import, division, print_function)

import numpy as np
from scipy.interpolate import interp1d
from lmfit import Model, models


class TabulatedModel(Model):
    """fitting the tabulated Model to some arbitrary points

        Parameters
        ----------
        xs: :class:`~numpy:numpy.ndarray`
            given domain of the function, energy

        ys: :class:`~numpy:numpy.ndarray`
            given domain of the function, intensity

        Fitting parameters:
            - rescaling factor ``amplitude``
            - shift along the X-axis ``center``
    """

    def __init__(self, xs, ys, *args, **kwargs):
        x = xs.reshape(xs.size)
        y = ys.reshape(ys.size)
        y_at_xmin = y[np.argmin(x)]
        y_at_xmax = y[np.argmax(x)]
        self._interp = interp1d(x, y, fill_value=(y_at_xmin, y_at_xmax),
                                bounds_error=False, kind='linear')

        def interpolator(x, amplitude, center):
            return amplitude * self._interp(x - center)

        super(TabulatedModel, self).__init__(interpolator, *args, **kwargs)
        self.set_param_hint('amplitude', value=1.0)
        self.set_param_hint('center', value=0.0)

    def guess(self, data, x, **kwargs):

        """Guess starting values for the parameters of a model.

          Parameters
          ----------
          data: :class:`~numpy:numpy.ndarray`
                data to be fitted

          x: :class:`~numpy:numpy.ndarray`
                energy domain where the interpolation required

          kwargs : dict
                additional optional arguments, passed to model function.

          Returns
          -------
          :class:`~lmfit.parameter.Parameters`
                parameters with guessed values

        """
        params = self.make_params()

        def pset(param, value, min):
            params["%s%s" % (self.prefix, param)].set(value=value, min=min)

        x_at_max = x[models.index_of(data, max(data))]
        ysim = self.eval(x=x_at_max, amplitude=1, center=x_at_max)
        amplitude = max(data) / ysim
        pset("amplitude", amplitude, min=0.0)
        pset("center",  x_at_max, min=-1000)
        return models.update_param_vals(params, self.prefix, **kwargs)
