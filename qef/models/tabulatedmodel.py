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

        amp : float
            Integrated intensity of the curve

        cen : float
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
        def interpolator(x, amp, cen):
            return amp * self._interp(x - cen)

        super(TabulatedModel, self).__init__(interpolator, *args, **kwargs)

    def guess(self, x, data, **kwargs):
        params = self.make_params()

        def pset(param,value):
            params["%s%s" %(self.prefix, param)].set(value=value)
        pset("amp", sum(data) * (max(x)-min(x))/len(x))
        pset("cen", x[models.index_of(data, max(data))])
        return models.update_param_vals(params, self.prefix, **kwargs)


