#!/usr/bin/env python
# -*- coding: utf-8 -*-\

from __future__ import (absolute_import, division, print_function)

import os
import pytest
import lmfit
import ipywidgets as ipyw
import qef.widgets.parameter as pqef


class TestParameterWidget(object):

    def test_init(self):
        p = pqef.ParameterWidget()
        assert p.nomin.value is True
        assert p.min.value == -float('inf')


def test_create_facade(widgets_fix):
    # Composite widget with all components and default names
    p = pqef.ParameterWidget()
    f = pqef.create_facade(p)
    for name in pqef.ParameterCallbacksMixin.widget_names:
        assert f[name] is p.__dict__[name]

    # Composite widget with all components and custom names
    p = widgets_fix['CustomParm']()
    mapping = widgets_fix['mapping']
    f = pqef.create_facade(p, mapping=mapping)
    for name in pqef.ParameterCallbacksMixin.widget_names:
        assert f[name] is p.__dict__[mapping[name]]

    # Single widget with default target
    p = ipyw.FloatSlider()
    f = pqef.create_facade(p)
    assert f['value'] is p

    # Single widget with custom target
    p = ipyw.Checkbox(value=True)
    f = pqef.create_facade(p, 'vary')
    assert f['vary'] is p


def test_add_widget_callbacks(widgets_fix):
    p = widgets_fix['CustomParm']()
    p.facade = pqef.create_facade(p, mapping=widgets_fix['mapping'])
    pqef.add_widget_callbacks(p)
    for name in ('nomin', 'min', 'value', 'nomax', 'max', 'vary', 'expr'):
        assert hasattr(p, name + '_value_change') is True


class TestParameterCallbacksMixin(object):

    def test_callbacks(self, widgets_fix):
        p = widgets_fix['CustomParm']()
        p.facade = pqef.create_facade(p, mapping=widgets_fix['mapping'])
        pqef.add_widget_callbacks(p)

        p.facade['min'].value = -1.0
        assert p.facade['nomin'].value is False  # nomin notified of min change
        p.facade['nomin'].value = True
        assert p.facade['min'].value == -p.inf  # min was notified of nomin change

        p.facade['value'].value = 0.5
        p.facade['min'].value = 0.0
        assert p.facade['value'].value == 0.5  # value not updated
        p.facade['min'].value = 1.0
        assert p.facade['value'].value == p.facade['min'].value  # value was notified of min change

        p.facade['max'].value = 10.0
        assert p.facade['nomax'].value is False  # nomax was notified of max change
        p.facade['nomax'].value = True
        assert p.facade['max'].value == p.inf  # max was notified of nomax change

        p.facade['min'].value = -1.0
        p.facade['value'].value = 0.0
        p.facade['max'].value = 1.0
        assert p.facade['value'].value == 0.0  # value not updated
        p.facade['max'].value = -0.5
        assert p.facade['value'].value == p.facade['max'].value  # value was notified of max change

        p.facade['min'].value = -1.0
        p.facade['value'].value = 0.0
        p.facade['max'].value = 1.0
        p.facade['min'].value = p.facade['max'].value + 1.0
        assert p.facade['min'].value == -1.0  # min value rejected
        p.facade['max'].value = p.facade['min'].value - 1.0
        assert p.facade['max'].value == 1.0  # max value rejected

        p.facade['min'].value = -1.0
        p.facade['value'].value = -2.0
        assert p.facade['value'].value == p.facade['min'].value  # value within bounds
        p.facade['max'].value = 1.0
        p.facade['value'].value = 2.0
        assert p.facade['value'].value == p.facade['max'].value  # value within bounds


class TestParameterWithTraits(object):

    def test_init(self):
        p = pqef.ParameterWithTraits(name='p', value=32.0)
        assert p.value == 32.0 and p.name == 'p'

    def test_repr(self):
        p = pqef.ParameterWithTraits(name='p', value=24)
        r = "<ParameterWithTraits <Parameter 'p', 24, bounds=[-inf:inf]>>"
        assert repr(p) == r

    def test_setattr(self):
        p = pqef.ParameterWithTraits(name='p', value=24)
        assert p.value == 24
        p.value = 42  # assignent to p.value
        assert p.value == 42
        assert p.tvalue == 42  # p.t_value is target of p.value
        p.tvalue = 24  # assignment to p.t_value
        assert p.tvalue == 24
        assert p.value == 24

    def test_set(self):
        p = lmfit.Parameter(name='p')
        p.set(expr='hello')
        p.set(value=42)
        p = pqef.ParameterWithTraits(name='p')
        p.set(value=42)
        assert p.tvalue == 42 and p.vary is True
        p.set(expr='hello')
        assert p.expr == 'hello' and p.texpr == 'hello'
        assert p.vary is False and p.tvary is False
        p.set(value=42)
        assert p.tvalue == 42 and p.vary is False
        assert p._expr is None and p.texpr is None

    def test_link_widget(self):
        p = pqef.ParameterWithTraits(name='p', value=24)
        w = ipyw.FloatSlider()
        p.link_widget(w)


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
