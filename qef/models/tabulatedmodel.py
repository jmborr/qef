import numpy as np
from scipy.interpolate import interp1d
from lmfit import Model, models
from lmfit.lineshapes import lorentzian
from matplotlib import pyplot as plt


class TabulatedModel (Model):
    """fitting the tabulated Model to some arbitrary points

        Parameters
        ----------
        xs: :class:`~numpy:numpy.ndarray`
            given domain of the function, energy

        ys: :class:`~numpy:numpy.ndarray`
            given domain of the function, intensity

        x: :class:`~numpy:numpy.ndarray`
            energy domain where the interpolation required

        amplitude : float
            peak intensity of the curve

        center : float
            position of the peak

        data: :class:`~numpy:numpy.ndarray`
            data to be fitted

        Returns
        -------
        :class:`~lmfit.parameter.Parameters`
            parameters with guessed values

        """

    def __init__(self, xs, ys, *args, **kwargs):
        self._interp = interp1d(xs, ys, fill_value='extrapolate', kind='cubic')

        def interpolator(x, amplitude, center):
            return amplitude * self._interp(x - center)

        super(TabulatedModel, self).__init__(interpolator, *args, **kwargs)

    def guess(self, data,x, **kwargs):
        params = self.make_params()

        def pset(param, value):
            params["%s%s" % (self.prefix, param)].set(value=value)

        x_at_max = x[models.index_of(data, max(data))]
        ysim = self.eval(x=x_at_max, amplitude=1, center=x_at_max)
        amplitude = max(data) / ysim
        pset("amplitude", amplitude )
        pset("center",  x_at_max)
        return models.update_param_vals(params, self.prefix, **kwargs)




