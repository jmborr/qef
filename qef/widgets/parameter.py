from __future__ import (absolute_import, division, print_function)

import lmfit
import traitlets
import numpy as np

class Simple(traitlets.HasTraits):
    t_value = traitlets.Float()


class ParameterWithTraits(lmfit.Parameter, traitlets.HasTraits):
    r"""Wrapper of lmfit.Parameter allows synchronization with ipywidgets
    """
    param_attr_names = ('_val', 'min', 'max', 'vary', '_expr')
    widget_trait_names = ('value', 'min', 'max', 'vary', 'expr')
    to_trait_prefix = 't'
    t_val = traitlets.Float(allow_none=True)
    tmin = traitlets.Float()
    tmax = traitlets.Float()
    tvary = traitlets.Bool()
    t_expr = traitlets.Unicode(allow_none=True)

    @classmethod
    def _to_trait(cls, key, origin='param'):
        other_key = key
        if origin == 'widget':
            # from widget to param

        return cls.to_trait_prefix + other_key


    @classmethod
    def _to_parm(cls, key):
        return key.replace(ParameterWithTraits.to_trait_prefix, '')

    def __init__(self, name=None, value=None, vary=True, min=-np.inf,
                 max=np.inf, expr=None, brute_step=None, user_data=None):
        kwargs = dict(name=name, value=value, vary=vary, min=min, max=max,
                      expr=expr, brute_step=brute_step, user_data=user_data)
        lmfit.Parameter.__init__(self, **kwargs)

    def __repr__(self):
        r"""String representation at debug level"""
        p_repr = super(ParameterWithTraits, self).__repr__()
        return '<ParameterWithTraits {}>'.format(p_repr)

    def __setattr__(self, key, value):
        if key in ParameterWithTraits.param_attr_names:
            lmfit.Parameter.__setattr__(self, key, value)
            other_key = ParameterWithTraits._to_trait(key)
            other_value = getattr(self, other_key)
            if value != other_value:  # prevent cycling
                traitlets.HasTraits.__setattr__(self, other_key, value)
        else:
            traitlets.HasTraits.__setattr__(self, key, value)
            other_key = ParameterWithTraits._to_parm(key)
            if other_key in ParameterWithTraits.param_attr_names:
                other_value = getattr(self, other_key)
                lmfit.Parameter.__setattr__(self, other_key, value)

    def link(self, widget, name):
        r"""bidirectional link with an ipywidget

        Parameters
        ----------
        w : ipywidget
            IPython widget containing one traitlet
        name : string
            traitlet name
        """
        key =
