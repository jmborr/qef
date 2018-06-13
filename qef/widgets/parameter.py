from __future__ import (absolute_import, division, print_function)

from collections import namedtuple
import lmfit
import traitlets
import ipywidgets as ipyw
import weakref

from qef.io import log_qef


class ParameterWidget(ipyw.Box):
    r"""One possible representation of a fitting parameter

    Parameters
    ----------
    show_header : Bool
        Hide or show names of the widget components `min`, `value`,...
    """

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

        # Layout
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


class ParameterCallbacksMixin(object):
    r"""Implement relationships between the different components of an
    ipywidget exposing all or some of the parameter attributes

    The methods in this Mixin expects attribute `facade`, a dictionary whose
    keys coincide with tuple `widget_names` and whose values are either None
    or references to ipython widgets"""

    inf = float('inf')
    widget_names = ('nomin', 'min', 'value', 'nomax', 'max', 'vary', 'expr')

    def validate_facade(self):
        r"""Ascertain that keys of facade attribute are contained in the
        set of widget names"""
        fs = set(self.facade.keys())
        assert set(ParameterCallbacksMixin.widget_names).issuperset(fs)

    def initialize_callbacks(self):
        r"""Register callbacks to sync widget components"""
        self.validate_facade()
        for widget_name in self.facade:
            widget = self.facade[widget_name]
            if widget is not None:
                callback = getattr(self, widget_name + '_value_change')
                widget.observe(callback, 'value', 'change')

    def nomin_value_change(self, change):
        r"""Set min to -infinity if nomin is checked"""
        if 'min' in self.facade and change.new is True:
            if self.facade['min'].value > -self.inf:  # prevent cycles
                self.facade['min'].value = -self.inf

    def min_value_change(self, change):
        r"""Notify other widgets if min changes.

        0. Reject change if min becomes bigger than max
        1. Uncheck nomin if new value is entered in min
        2. Update value.value if it becomes smaller than min.value"""
        if 'max' in self.facade and change.new > self.facade['max'].value:
            self.facade['min'].value = change.old  # reject change
        else:  # Notify other widgets
            if 'nomin' in self.facade and change.new > -self.inf:
                self.facade['nomin'].value = False
            if 'value' in self.facade and \
                change.new > self.facade['value'].value:
                self.facade['value'].value = change.new

    def nomax_value_change(self, change):
        r"""Set max to infinity if nomax is checked"""
        if 'max' in self.facade and change.new is True:
            if self.facade['max'].value < self.inf:  # prevent cycles
                self.facade['max'].value = self.inf

    def max_value_change(self, change):
        r"""Notify other widgets if min changes.

        0. Reject change if max becomes smaller than min
        1. Uncheck nomax if new value is entered in max
        2. Update value.value if it becomes bigger than max.value"""
        if 'min' in self.facade and change.new < self.facade['min'].value:
            self.facade['max'].value = change.old  # reject change
        else:  # Notify other widgets
            if 'nomax' in self.facade and change.new < self.inf:
                self.facade['nomax'].value = False
            if 'value' in self.facade and \
                change.new < self.facade['value'].value:
                self.facade['value'].value = change.new

    def value_value_change(self, change):
        r"""Validate value is within bounds. Otherwise set value as the
        closest bound value"""
        if 'min' in self.facade and change.new < self.facade['min'].value:
            self.facade['value'].value = self.facade['min'].value
        elif 'max' in self.facade and change.new > self.facade['max'].value:
            self.facade['value'].value = self.facade['max'].value

    def vary_value_change(self, change):
        r"""enable/disable editing of boundaries, value, and expression"""
        for name in ('nomin', 'min', 'value', 'nomax', 'max', 'expr'):
            if name in self.facade:
                self.facade['name'].disabled = not change.new

    def expr_value_change(self, change):
        r"""enable/disable boundaries and values"""
        if 'vary' in self.facade:
            self.facade['vary'].value = True if change.new == '' else False


def create_facade(widget, mapping=None):
    r"""Create facade dictionary where keys are standard widget names
    (ParameterCallbacksMixin.widget_names) and whose values are simple
    ipywidgets that control the fitting parameter attributes denoted by
    the standard widget names

    Parameters
    ----------
    widget: ipywidget
    mapping : str, dict, or None
        if `str`, mapping denotes the widget name to be associated with
        the widget. If `dict`, then `mapping` values are attribute names
        of `widget`, referencing the simple ipywidgets to be associated
        to standard  widget names. The widget names are the keys of `mapping`.
        If `None`, an inspection of `widget` attributes will be performed,
        looking for names that coincide with standard widget names. If the
        inspection is unsuccessful, the widget will be associated with the
        standard widget name 'value' to represent the values taken by the
        fitting parameter.

    Returns
    -------
    facade : dict
    """
    names = ParameterCallbacksMixin.widget_names  # expected widget names
    if mapping is not None:
        if isinstance(mapping, str) and mapping in names:
            # subscribing a non-composite widget
            facade = {mapping: widget}
        elif isinstance(mapping, dict):
            k = set(mapping.keys())
            if k & set(names) != k:
                msg = 'mapping contains invalid widget names'
                raise KeyError(msg)
            facade = {name: widget.__dict__[wn]
                      for name, wn in mapping.items()}
    else:  # inspection
        facade = {name: widget.__dict__[name] for name in names
                  if name in widget.__dict__}
        if bool(facade) is False:
            facade = {'value': widget}
    return facade


def add_widget_facade(widget, mapping=None):
    r"""Create facade dictionary where keys are standard widget names
    (ParameterCallbacksMixin.widget_names) and whose values are simple
    ipywidgets that control the fitting parameter attributes denoted by
    the standard widget names. This dictionary is added to the input
    widget as an attribute.

    Parameters
    ----------
    widget: ipywidget
    mapping : str, dict, or None
        if `str`, mapping denotes the widget name to be associated with
        the widget. If `dict`, then `mapping` values are attribute names
        of `widget`, referencing the simple ipywidgets to be associated
        to standard  widget names. The widget names are the keys of `mapping`.
        If `None`, an inspection of `widget` attributes will be performed,
        looking for names that coincide with standard widget names. If the
        inspection is unsuccessful, the widget will be associated with the
        standard widget name 'value' to represent the values taken by the
        fitting parameter.

    Returns
    -------
    widget : ipywidget
        returns input widget
    """
    widget.facade = create_facade(widget, mapping=mapping)
    return widget


def add_widget_callbacks(widget, mapping=None):
    r"""Extend the widget's type with ParameterCallbacksMixin

    Parameters
    ----------
    widget: ipywidget
    mapping : str, dict, or None
        if `str`, mapping denotes the widget name to be associated with
        the widget. If `dict`, then `mapping` values are attribute names
        of `widget`, referencing the simple ipywidgets to be associated
        to standard  widget names. The widget names are the keys of `mapping`.
        If `None`, an inspection of `widget` attributes will be performed,
        looking for names that coincide with standard widget names. If the
        inspection is unsuccessful, the widget will be associated with the
        standard widget name 'value' to represent the values taken by the
        fitting parameter.
    """
    base_class = widget.__class__
    widget.__class__ = type(base_class.__name__,
                            (base_class, ParameterCallbacksMixin), {})
    widget.initialize_callbacks()


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
    param_attrs = ('_val', 'min', 'max', 'vary', '_expr')
    param_features = ('value', 'min', 'max', 'vary', 'expr')
    trait_names = ('tvalue', 'tmin', 'tmax', 'tvary', 'texpr')
    #: :class:# `~traitlets.Float` trailet wrapping
    #: :class:`~lmfit.parameter.Parameter` attribute `value`
    tvalue = traitlets.Float(allow_none=True)
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
    texpr = traitlets.Unicode(allow_none=True)

    @classmethod
    def feature_to_trait(cls, feature):
        r"""From :class:`~lmfit.parameter.Parameter` feature name to
        :class:`~traitlets.TraitType` name"""
        try:
            return cls.trait_names[cls.param_features.index(feature)]
        except:
            raise KeyError('{} is not a parameter feature'.format(feature))

    @classmethod
    def attr_to_trait(cls, attr):
        r"""From :class:`~lmfit.parameter.Parameter` attribute name to
        :class:`~traitlets.TraitType` name"""
        try:
            return cls.trait_names[cls.param_attrs.index(attr)]
        except:
            raise KeyError('{} is not a parameter feature'.format(attr))

    @classmethod
    def trait_to_attr(cls, name):
        r"""From :class:`~traitlets.TraitType` name to
        :class:`~lmfit.parameter.Parameter` attribute name"""
        try:
            return cls.param_attrs[cls.trait_names.index(name)]
        except:
            raise KeyError('{} is not a valid trait'.format(name))

    def __init__(self, name=None, value=None, vary=True, min=-float('inf'),
                 max=float('inf'), expr=None, brute_step=None, user_data=None):
        kwargs = dict(name=name, value=value, vary=vary, min=min, max=max,
                      expr=expr, brute_step=brute_step, user_data=user_data)
        lmfit.Parameter.__init__(self, **kwargs)
        self._widget_links = weakref.WeakSet()

    def __repr__(self):
        r"""String representation at debug level"""
        p_repr = super(ParameterWithTraits, self).__repr__()
        return '<ParameterWithTraits {}>'.format(p_repr)

    def __setattr__(self, key, value):
        r"""Setting attributes making sure :class:`~lmfit.parameter.Parameter`
        attributes and :class:`~traitlets.TraitType` stay in sync"""
        if key in ParameterWithTraits.param_attrs:
            # attribute of Parameter
            lmfit.Parameter.__setattr__(self, key, value)
            other_key = ParameterWithTraits.attr_to_trait(key)
            other_value = getattr(self, other_key)
            if value != other_value:  # prevent cycling
                traitlets.HasTraits.__setattr__(self, other_key, value)
        else:
            # attribute of HasTraits
            traitlets.HasTraits.__setattr__(self, key, value)
            if key in ParameterWithTraits.trait_names:
                other_key = ParameterWithTraits.trait_to_attr(key)
                other_value = getattr(self, other_key)
                if value != other_value:  # prevent cycling
                    lmfit.Parameter.__setattr__(self, other_key, value)

    def link_widget(self, widget, mapping=None):
        r"""

        Parameters
        ----------
        widget: ipywidget
        mapping : str, dict, or None
            if `str`, mapping denotes the widget name to be associated with
            the widget. If `dict`, then `mapping` values are attribute names
            of `widget`, referencing the simple ipywidgets to be associated
            to standard  widget names. The widget names are the keys of `mapping`.
            If `None`, an inspection of `widget` attributes will be performed,
            looking for names that coincide with standard widget names. If the
            inspection is unsuccessful, the widget will be associated with the
            standard widget name 'value' to represent the values taken by the
            fitting parameter.
        """
        add_widget_facade(widget, mapping=mapping)
        add_widget_callbacks(widget, mapping=mapping)
        for pn, w in widget.facade.items():
            tname = self.feature_to_trait(pn)
            if w not in [l.target[0] for l in self._widget_links]:
                l = traitlets.link((self, tname), (w, 'value'))
                self._widget_links.add(l)
