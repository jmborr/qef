from __future__ import (absolute_import, division, print_function)

from qef.models.tabulatedmodel import TabulatedModel

class TabulatedResolutionModel(TabulatedModel):
    r"""Interpolator of resolution data with no fit parameters

    Parameters
    ----------
    xs: :class:`~numpy:numpy.ndarray`
        given domain of the function, energy

    ys: :class:`~numpy:numpy.ndarray`
        given domain of the function, intensity
    """

    def __init__(self, xs, ys, *args, **kwargs):
        super(TabulatedResolutionModel, self).__init__(xs, ys, *args, **kwargs)
        self.set_param_hint('amplitude', value=1.0, vary=False)
        self.set_param_hint('center', value=0.0, vary=False)
