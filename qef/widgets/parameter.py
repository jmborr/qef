from __future__ import (absolute_import, division, print_function)

import lmfit
import traitlets
import ipywidgets as ipyw
import numpy as np


class ParameterWithTraits(lmfit.Parameter, traitlets.HasTraits):
    r"""Wrapper of :class:`~lmfit.parameter.Parameter` with
    :class:`~traitlets.TraitType` allows synchronization with ipywidgets

    Same signature for initialization as that of
    :class:`~lmfit.parameter.Parameter`.

    Parameters
    ----------
    name : str, optional
        Name of the Parameter.
    value : float, optional
        Numerical Parameter value.
    vary : bool, optional
        Whether the Parameter is varied during a fit (default is True).
    min : float, optional
        Lower bound for value (default is `-numpy.inf`, no lower bound).
    max : float, optional
        Upper bound for value (default is `numpy.inf`, no upper bound).
    expr : str, optional
        Mathematical expression used to constrain the value during the fit.
    brute_step : float, optional
        Step size for grid points in the `brute` method.
    user_data : optional
        User-definable extra attribute used for a Parameter.
    """
    #: :class:`~lmfit.parameter.Parameter` attribute  names
    param_attr_names = ('_val', 'min', 'max', 'vary', '_expr')
    to_trait_prefix = 't'
    #: :class:`~traitlets.Float` trailet wrapping
    #: :class:`~lmfit.parameter.Parameter` attribute `value`
    t_val = traitlets.Float(allow_none=True)
    #: :class:`~traitlets.Float` trailet wrapping
    #: :class:`~lmfit.parameter.Parameter` attribute `_val`
    tmin = traitlets.Float()
    #: :class:`~traitlets.Float` trailet wrapping
    #: :class:`~lmfit.parameter.Parameter` attribute `min`
    tmax = traitlets.Float()
    #: :class:`~traitlets.Bool` trailet wrapping
    #: :class:`~lmfit.parameter.Parameter` attribute `vary`
    tvary = traitlets.Bool()
    #: :class:`~traitlets.Unicode` trailet wrapping
    #: :class:`~lmfit.parameter.Parameter` attribute `expr`
    t_expr = traitlets.Unicode(allow_none=True)

    @classmethod
    def _to_trait(cls, key):
        r"""From :class:`~lmfit.parameter.Parameter` attribute name to
        :class:`~traitlets.TraitType` name"""
        return cls.to_trait_prefix + key

    @classmethod
    def _to_parm(cls, key):
        r"""From :class:`~traitlets.TraitType` name to
        :class:`~lmfit.parameter.Parameter` attribute name"""
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
        r"""Setting attributes making sure :class:`~lmfit.parameter.Parameter`
        attributes and :class:`~traitlets.TraitType` stay in sync"""
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


class ParameterWidget(ipyw.Box):
    r"""One possible representation of a fitting parameter

    Parameters
    ----------
    show_header : Bool
        Hide or show names of the widget components `min`, `value`,...
    """
    inf = float('inf')

    def __init__(self, show_header=True):
        def el_ly(w):
            return dict(width='{}px'.format(w), margin='0px')

        # minimum block
        self.nomin = ipyw.Checkbox(value=True, layout=el_ly(125))
        self.min = ipyw.FloatText(value=-self.inf, layout=el_ly(165))
        box_ly = dict(border='1px solid black', display='flex', margin='0px',
                      flex_flow='row', width='290px')
        self.minbox = ipyw.Box([self.nomin, self.min], layout=box_ly)

        # value element
        self.value = ipyw.FloatText(value=0, layout=el_ly(170))

        # maximum block
        self.nomax = ipyw.Checkbox(value=True, layout=el_ly(125))
        self.max = ipyw.FloatText(value=self.inf, layout=el_ly(165))
        self.maxbox = ipyw.Box([self.nomax, self.max], layout=box_ly)

        # constraints block
        self.vary = ipyw.Checkbox(value=True, layout=el_ly(125))
        self.expr = ipyw.Text(value='', continuous_update=False,
                              layout=el_ly(275))

        # array elements in an horizontal block
        self.elements = [self.minbox, self.value, self.maxbox,
                         self.vary, self.expr]

        # Header labels
        self.header = None
        if show_header is True:
            d_lbs = (('-inf', 125), ('min', 165), ('value', 170),
                     ('inf', 125), ('max', 165), ('vary', 125),
                     ('expression', 275))
            l_lbs = [ipyw.Label(k, layout=el_ly(v)) for (k, v) in d_lbs]
            box_ly = dict(display='flex', margin='0px', border='solid')
            self.header = ipyw.HBox(l_lbs, layout=box_ly)

        self.initialize_callbacks()

        if self.header is None:
            box_ly = dict(display='flex', margin='0px', border='solid',
                          flex_flow='row')
            super(ParameterWidget, self).__init__(self.elements, layout=box_ly)
        else:
            box_ly = dict(display='flex', margin='0px', border='solid')
            b_els = ipyw.HBox(self.elements, layout=box_ly)
            box_ly.update({'flex_flow': 'column'})
            super(ParameterWidget, self).__init__([self.header, b_els],
                                                  layout=box_ly)

    def initialize_callbacks(self):
        r"""Register callbacks to sync component widgets"""
        self.nomin.observe(self.nomin_changed, 'value', 'change')
        self.min.observe(self.min_changed, 'value', 'change')
        self.nomax.observe(self.nomax_changed, 'value', 'change')
        self.max.observe(self.max_changed, 'value', 'change')
        self.value.observe(self.value_changed, 'value', 'change')
        self.vary.observe(self.vary_changed, 'value', 'change')
        self.expr.observe(self.expr_changed, 'value', 'change')

    def nomin_changed(self, change):
        r"""Set min to -infinity if nomin is checked"""
        if change.new is True:
            if self.min.value > -self.inf:  # prevent cycles
                self.min.value = -self.inf

    def min_changed(self, change):
        r"""Notify other widgets if min changes.

        0. Reject change if min becomes bigger than max
        1. Uncheck nomin if new value is entered in min
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
        r"""Notify other widgets if min changes.

        0. Reject change if max becomes smaller than min
        1. Uncheck nomax if new value is entered in max
        2. Update value.value if it becomes bigger than max.value"""
        if change.new < self.min.value:  # Validate bounds
            self.max.value = change.old  # reject change
        else:  # Notify other widgets
            if change.new < self.inf:
                self.nomax.value = False
            if change.new < self.value.value:
                self.value.value = change.new

    def value_changed(self, change):
        r"""Validate value is within bounds. Otherwise set value as the
        closest bound value"""
        if change.new < self.min.value:
            self.value.value = self.min.value
        elif change.new > self.max.value:
            self.value.value = self.max.value

    def vary_changed(self, change):
        r"""enable/disable eidtin of boundaries, value, and expression"""
        for w in (self.nomin, self.min, self.value, self.nomax, self.max,
                  self.expr):
            w.disabled = not change.new

    def expr_changed(self, change):
        r"""enable/disable boundaries and values"""
        self.vary.value = True if change.new == '' else False
