import numpy as np
from scipy.interpolate import interp1d
from lmfit import Model, models
from lmfit.lineshapes import lorentzian
from matplotlib import pyplot as plt



class TabulatedModel (Model):
    """fitting the tabulated Model to some arbitrary points

        Parameters
        ----------
        x_orig :array of floats
            given domain of the function, energy
        x :array of floats
            domain of the function where interpolation is desired
        shift: floats
            shift of the function
        amp : float
            Integrated intensity of the curve
        cen : float
            position of the peak
        sig : float
            broadening of the curve
        data: array of floats
            data to be fitted

        """

    def __init__(self, x_orig, shift, *args, **kwargs):
        def interpolator( x, amp, cen, sig):
            y = lorentzian(x_orig-shift, amp, cen, sig)
            return interp1d(x_orig, y, fill_value='interpolate', kind='cubic')(x)

        super(TabulatedModel, self).__init__(interpolator, *args, **kwargs)

    def guess(self, x, data, sig, **kwargs):
        params = self.make_params()

        def pset(param,value):
            params["%s%s" %(self.prefix, param)].set(value=value)
        pset("amp", sum(data) * (max(x)-min(x))/len(x))
        pset("cen", x[models.index_of(data, max(data))])
        pset("sig", sig)
        return models.update_param_vals(params, self.prefix, **kwargs)


