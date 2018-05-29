from __future__ import (absolute_import, division, print_function)

import lmfit
import traitlets
import ipywidgets as ipyw
import numpy as np


class ParameterWithTraits(lmfit.Parameter, traitlets.HasTraits):
    r"""Wrapper of lmfit.Parameter allows synchronization with ipywidgets
    """
    param_attr_names = ('_val', 'min', 'max', 'vary', '_expr')
    to_trait_prefix = 't'
    t_val = traitlets.Float(allow_none=True)
    tmin = traitlets.Float()
    tmax = traitlets.Float()
    tvary = traitlets.Bool()
    t_expr = traitlets.Unicode(allow_none=True)

    @classmethod
    def _to_trait(cls, key):
        return cls.to_trait_prefix + key

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
        r"""Setting attributes making sure Parameter attributes and
        traitlets stay in sync"""
        if key in ParameterWithTraits.param_attr_names:
            # attribute of Parameter
            lmfit.Parameter.__setattr__(self, key, value)
            other_key = ParameterWithTraits._to_trait(key)
            other_value = getattr(self, other_key)
            if value != other_value:  # prevent cycling
                traitlets.HasTraits.__setattr__(self, other_key, value)
        else:
            # attribute of HasTraits
            traitlets.HasTraits.__setattr__(self, key, value)
            other_key = ParameterWithTraits._to_parm(key)
            if other_key in ParameterWithTraits.param_attr_names:
                other_value = getattr(self, other_key)
                if value != other_value:  # prevent cycling
                    lmfit.Parameter.__setattr__(self, other_key, value)


class ParameterWidget(ipyw.HBox):
    r"""One possible representation of a fitting parameter"""
    inf = float('inf')

    def __init__(self):
        self.value = ipyw.FloatText(description='value:', value=0, allow_none=True)
        self.nomin = ipyw.Checkbox(description='-inf', value=True)
        self.min = ipyw.FloatText(description='min:', value=-self.inf)
        self.nomax = ipyw.Checkbox(description='inf', value=True)
        self.max = ipyw.FloatText(description='max:', value=self.inf)
        self.vary = ipyw.Checkbox(description='vary', value=True)
        self.expr = ipyw.Text(description='constrain:', value='')
        self.elements = [self.nomin, self.min, self.value,
                         self.nomax, self.max, self.vary, self.expr]
        self.initialize_callbacks()
        self.initialize_layouts()
        super(ParameterWidget, self).__init__(self.elements)

    def initialize_callbacks(self):
        r"""Register callbacks to sync elements"""
        self.nomin.observe(self.nomin_changed, 'value', 'change')
        self.min.observe(self.min_changed, 'value', 'change')
        self.nomax.observe(self.nomax_changed, 'value', 'change')
        self.max.observe(self.max_changed, 'value', 'change')
        self.value.observe(self.value_changed, 'value', 'change')

    def nomin_changed(self, change):
        r"""Set min to -infinity if nomin is checked"""
        if change.new is True:
            if self.min.value > -self.inf:  # prevent cycles
                self.min.value = -self.inf

    def min_changed(self, change):
        r"""1. Uncheck nomin if new value is entered in min
        2. Update value.value if it becomes smaller than min.value"""
        if change.new > self.max.value:  # Validate bounds
            self.min.value = change.old  # reject change
        else:  # Notify other widgets
            if change.new > -self.inf:
                self.nomin.value = False
            if change.new > self.value.value:
                self.value.value = change.new

    def nomax_changed(self, change):
        r"""Set max to infinity if nomax is checked"""
        if change.new is True:
            if self.max.value < self.inf:  # prevent cycles
                self.max.value = self.inf

    def max_changed(self, change):
        r"""1. Uncheck nomax if new value is entered in max
        2. Update value.value if it becomes bigger than max.value"""
        if change.new < self.min.value:  # Validate bounds
            self.max.value = change.old  # reject change
        else:  # Notify other widgets
            if change.new < self.inf:
                self.nomax.value = False
            if change.new < self.value.value:
                self.value.value = change.new

    def value_changed(self, change):
        r"""Validate value within bounds"""
        if change.new < self.min.value:
            self.value.value = self.min.value
        elif change.new > self.max.value:
            self.value.value = self.max.value

    def initialize_layouts(self):
        r"""Prettify the look"""
