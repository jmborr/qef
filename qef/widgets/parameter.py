from __future__ import (absolute_import, division, print_function)

import lmfit
import traitlets
import ipywidgets as ipyw
import weakref
from qef.io import log_qef


class ParameterWidget(ipyw.Box):
    r"""One possible representation of a fitting parameter.
    Inherits from `ipywidgets.widgets.widget_box.Box <https://github.com/jupyter-widgets/ipywidgets/blob/v7.0.0a1/ipywidgets/widgets/widget_box.py#L18>`_

    Parameters
    ----------
    show_header : Bool
        Hide or show names of the widget components `min`, `value`,...
    """  # noqa: E501

    def __init__(self, show_header=True):
        def el_ly(w):
            return dict(width='{}px'.format(w), margin='0px')

        # minimum block
        self.nomin = ipyw.Checkbox(value=True, layout=el_ly(125))
        self.min = ipyw.FloatText(value=-float('inf'), layout=el_ly(165))
        box_ly = dict(border='1px solid black', display='flex', margin='0px',
                      flex_flow='row', width='290px')
        self.minbox = ipyw.Box([self.nomin, self.min], layout=box_ly)

        # value element
        self.value = ipyw.FloatText(value=0, layout=el_ly(170))

        # maximum block
        self.nomax = ipyw.Checkbox(value=True, layout=el_ly(125))
        self.max = ipyw.FloatText(value=float('inf'), layout=el_ly(165))
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

    The methods in this Mixin expects attribute :code:`facade`,
    a dictionary whose keys coincide with tuple
    :const:`~qef.widgets.parameter.ParameterCallbacksMixin.widget_names`
    and whose values are either :code:`None` or references to ipywidgets.
    Attribute :code:`facade` can be created with
    function :func:`~qef.widgets.parameter.add_widget_facade`."""

    #: Representation of infinity value
    inf = float('inf')
    widget_names = ('nomin', 'min', 'value', 'nomax', 'max', 'vary', 'expr')

    def validate_facade(self):
        r"""Ascertain that keys of :code:`facade` attribute are contained in
        :const:`~qef.widgets.parameter.ParameterCallbacksMixin.widget_names`"""
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
        r"""Set :code:`min` to :math:`-\infty` if :code:`nomin` is checked"""
        if 'min' in self.facade and change.new is True:
            if self.facade['min'].value > -self.inf:  # prevent cycles
                self.facade['min'].value = -self.inf

    def min_value_change(self, change):
        r"""Notify other widgets if :code:`min` changes.

        0. Reject change if :code:`min` becomes bigger than :code:`max`
        1. Uncheck :code:`nomin` if new value is entered in :code:`min`
        2. Update :code:`value.value` if it becomes smaller than
           :code:`min.value`"""
        f = self.facade
        if 'max' in f and change.new > f['max'].value:
            f['min'].value = change.old  # reject change
        else:  # Notify other widgets
            if 'nomin' in f and change.new > -self.inf:
                f['nomin'].value = False
            if 'value' in f and change.new > f['value'].value:
                    f['value'].value = change.new

    def nomax_value_change(self, change):
        r"""Set :code:`max` to :math:`\infty` if :code:`nomax` is checked"""
        if 'max' in self.facade and change.new is True:
            if self.facade['max'].value < self.inf:  # prevent cycles
                self.facade['max'].value = self.inf

    def max_value_change(self, change):
        r"""Notify other widgets if :code:`min` changes.

        0. Reject change if :code:`max` becomes smaller than :code:`min`
        1. Uncheck :code:`nomax` if new value is entered in :code:`max`
        2. Update :code:`value.value` if it becomes bigger than
        :code:`max.value`"""
        f = self.facade
        if 'min' in f and change.new < f['min'].value:
            f['max'].value = change.old  # reject change
        else:  # Notify other widgets
            if 'nomax' in f and change.new < self.inf:
                f['nomax'].value = False
            if 'value' in f and change.new < f['value'].value:
                    f['value'].value = change.new

    def value_value_change(self, change):
        r"""Validate :code:`value` is within bounds. Otherwise set
        :code:`value` as the closest bound value"""
        if 'min' in self.facade and change.new < self.facade['min'].value:
            self.facade['value'].value = self.facade['min'].value
        elif 'max' in self.facade and change.new > self.facade['max'].value:
            self.facade['value'].value = self.facade['max'].value

    def vary_value_change(self, change):
        r"""enable/disable editing of :code:`min`, :code:`max`, :code:`value`,
        and :code:`expr`"""
        for name in ('nomin', 'min', 'value', 'nomax', 'max', 'expr'):
            if name in self.facade:
                self.facade['name'].disabled = not change.new

    def expr_value_change(self, change):
        r"""enable/disable :code:`min`, :code:`max`, and :code:`value`"""
        if 'vary' in self.facade:
            self.facade['vary'].value = True if change.new == '' else False


def create_facade(widget, mapping=None):
    r"""Create :code:`facade` dictionary where keys are standard
    :const:`~qef.widgets.parameter.ParameterCallbacksMixin.widget_names`
    and whose values are simple ipywidgets that control the fitting
    parameter attributes denoted by the standard
    :const:`~qef.widgets.parameter.ParameterCallbacksMixin.widget_names`.


    Parameters
    ----------
    widget: `ipywidgets.widgets.widget.Widget <https://github.com/jupyter-widgets/ipywidgets/blob/v7.0.0a1/ipywidgets/widgets/widget.py#L238>`_
    mapping : str, dict, or None
        if `str`, mapping denotes the widget name to be associated with
        the widget. If `dict`, then `mapping` values are attribute names
        of `widget`, referencing the simple ipywidgets to be associated
        to standard  widget names. The widget names are the keys of `mapping`.
        If :code:`None`, an inspection of `widget` attributes will be performed,
        looking for names that coincide with standard widget names. If the
        inspection is unsuccessful, the widget will be associated with the
        standard widget name 'value' to represent the values taken by the
        fitting parameter.

    Returns
    -------
    facade : dict
    """  # noqa: E501
    names = ParameterCallbacksMixin.widget_names  # expected widget names
    if mapping is not None:
        if isinstance(mapping, str) and mapping in names:
            # subscribing a non-composite widget
            facade = {mapping: widget}
        elif isinstance(mapping, dict):
            k = set(mapping.keys())
            if k & set(names) != k:
                msg = 'mapping contains invalid widget names'
                log_qef.error(msg)
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
    r"""Create :code:`facade` dictionary where keys are standard
    :const:`~qef.widgets.parameter.ParameterCallbacksMixin.widget_names`
    and whose values are simple ipywidgets that control the fitting
    parameter attributes denoted by the standard
    :const:`~qef.widgets.parameter.ParameterCallbacksMixin.widget_names`.
    This dictionary is added to the input widget as an attribute.

    Parameters
    ----------
    widget: `ipywidgets.widgets.widget.Widget <https://github.com/jupyter-widgets/ipywidgets/blob/v7.0.0a1/ipywidgets/widgets/widget.py#L238>`_
    mapping : str, dict, or None
        if `str`, mapping denotes the widget name to be associated with
        the widget. If `dict`, then `mapping` values are attribute names
        of `widget`, referencing the simple ipywidgets to be associated
        to standard  widget names. The widget names are the keys of `mapping`.
        If :code:`None`, an inspection of `widget` attributes will be performed,
        looking for names that coincide with standard widget names. If the
        inspection is unsuccessful, the widget will be associated with the
        standard widget name 'value' to represent the values taken by the
        fitting parameter.

    Returns
    -------
    widget : :class:`~ipywidgets:ipywidgets.widgets.widget.Widget`
        Reference to input widget
    """  # noqa: E501
    widget.facade = create_facade(widget, mapping=mapping)
    return widget


def add_widget_callbacks(widget, mapping=None):
    r"""Extend the widget's type with
    :class:`~qef.widgets.parameter.ParameterCallbacksMixin`

    Parameters
    ----------
    widget: `ipywidgets.widgets.widget.Widget <https://github.com/jupyter-widgets/ipywidgets/blob/v7.0.0a1/ipywidgets/widgets/widget.py#L238>`_
    mapping : str, dict, or None
        if `str`, :code:`mapping` denotes the widget name to be associated with
        the widget. If `dict`, then :code:`mapping` values are attribute names
        of `widget`, referencing the simple ipywidgets to be associated
        to standard
        :const:`~qef.widgets.parameter.ParameterCallbacksMixin.widget_names`.
        The widget names are the keys of :code:`mapping`.
        If :code:`None`, an inspection of `widget` attributes will be
        performed, looking for names that coincide with standard
        :const:`~qef.widgets.parameter.ParameterCallbacksMixin.widget_names`.
        If the inspection is unsuccessful, the widget will be associated
        with the standard widget name 'value' to represent the values taken
        by the fitting parameter.
    """  # noqa: E501
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
    #: :class:`~lmfit.parameter.Parameter` feature  names
    param_features = ('value', 'min', 'max', 'vary', 'expr')
    #: :class:`~traitlets.TraitType` instances in sync with
    #: :class:`~lmfit.parameter.Parameter` attributes
    trait_names = ('tvalue', 'tmin', 'tmax', 'tvary', 'texpr')
    #: :class:`~traitlets.Float` trailet wrapping
    #: :class:`~lmfit.parameter.Parameter` attribute :code:`value`
    tvalue = traitlets.Float(allow_none=True)
    #: :class:`~traitlets.Float` trailet wrapping
    #: :class:`~lmfit.parameter.Parameter` attribute :code:`_val`
    tmin = traitlets.Float()
    #: :class:`~traitlets.Float` trailet wrapping
    #: :class:`~lmfit.parameter.Parameter` attribute :code:`min`
    tmax = traitlets.Float()
    #: :class:`~traitlets.Bool` trailet wrapping
    #: :class:`~lmfit.parameter.Parameter` attribute :code:`vary`
    tvary = traitlets.Bool()
    #: :class:`~traitlets.Unicode` trailet wrapping
    #: :class:`~lmfit.parameter.Parameter` attribute :code:`_expr`
    texpr = traitlets.Unicode(allow_none=True)

    @classmethod
    def feature_to_trait(cls, feature):
        r"""From :class:`~lmfit.parameter.Parameter` feature name to
        :class:`~traitlets.TraitType` name"""
        try:
            return cls.trait_names[cls.param_features.index(feature)]
        except KeyError:
            msg = '{} is not a parameter feature'.format(feature)
            log_qef.error(msg)
            raise KeyError(msg)

    @classmethod
    def attr_to_trait(cls, attr):
        r"""From :class:`~lmfit.parameter.Parameter` attribute name to
        :class:`~traitlets.TraitType` name"""
        try:
            return cls.trait_names[cls.param_attrs.index(attr)]
        except KeyError:
            msg = '{} is not a parameter feature'.format(attr)
            log_qef.error(msg)
            raise KeyError(msg)

    @classmethod
    def trait_to_attr(cls, name):
        r"""From :class:`~traitlets.TraitType` name to
        :class:`~lmfit.parameter.Parameter` attribute name"""
        try:
            return cls.param_attrs[cls.trait_names.index(name)]
        except KeyError:
            msg = '{} is not a valid trait'.format(name)
            log_qef.error(msg)
            raise KeyError(msg)

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
        r"""Link the value of a single ipywidget to one trait, or the values
        of the element widgets of a composite ipywidget to different traits.
        The specific traits can be specified with the :code:`mapping` argument.

        Parameters
        ----------
        widget: `ipywidgets.widgets.widget.Widget <https://github.com/jupyter-widgets/ipywidgets/blob/v7.0.0a1/ipywidgets/widgets/widget.py#L238>`_
        mapping : str, dict, or None
            if `str`, :code:`mapping` denotes the widget name to be associated
            with the widget. If `dict`, then :code:`mapping` values are
            attribute names of `widget`, referencing the simple ipywidgets to
            be associated to standard
            :const:`~qef.widgets.parameter.ParameterCallbacksMixin.widget_names`.
            The widget names are the keys of :code:`mapping`. If :code:`None`,
            an inspection of `widget` attributes will be performed,
            looking for names that coincide with standard
            :const:`~qef.widgets.parameter.ParameterCallbacksMixin.widget_names`.
            If the inspection is unsuccessful, the widget will be associated
            with the standard widget name 'value' to represent the values
            taken by the fitting parameter.
        """  # noqa: E501
        add_widget_facade(widget, mapping=mapping)
        add_widget_callbacks(widget, mapping=mapping)
        for pn, w in widget.facade.items():
            tname = self.feature_to_trait(pn)
            if w not in [l.target[0] for l in self._widget_links]:
                lnk = traitlets.link((self, tname), (w, 'value'))
                self._widget_links.add(lnk)
