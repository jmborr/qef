#!/usr/bin/env python
# -*- coding: utf-8 -*-\

from __future__ import (absolute_import, division, print_function)

import os
import numpy as np
import pytest
import lmfit
from qef.widgets.parameter import ParameterWithTraits, ParameterWidget


class TestParameterWithTraits(object):

    def test_init(self):
        p = ParameterWithTraits(name='p', value=32.0)
        assert p.value == 32.0 and p.name == 'p'

    def test_setattr(self):
        p = ParameterWithTraits(name='p', value=24)
        assert p.value == 24
        p.value = 42  # assignent to p.value
        assert p.value == 42
        assert p.t_val == 42  # p.t_value is target of p.value
        p.t_val = 24  # assignment to p.t_value
        assert p.t_val == 24
        assert p.value == 24

    def test_set(self):
        p = lmfit.Parameter(name='p')
        p.set(expr='hello')
        p.set(value=42)
        p = ParameterWithTraits(name='p')
        p.set(value=42)
        assert p.t_val == 42 and p.vary is True
        p.set(expr='hello')
        assert p.expr == 'hello' and p.t_expr == 'hello'
        assert p.vary is False and p.tvary is False
        p.set(value=42)
        assert p.t_val == 42 and p.vary is False
        assert p._expr is None and p.t_expr is None


class TestParameterWidget(object):

    def test_init(self):
        p = ParameterWidget()
        assert p.nomin.value is True
        assert p.min.value == -p.inf

    def test_initialize(self):
        p = ParameterWidget()

        p.min.value = -1.0
        assert p.nomin.value is False  # nomin was notified of min change
        p.nomin.value = True
        assert p.min.value == -p.inf  # min was notified of nomin change

        p.value.value = 0.5
        p.min.value = 0.0
        assert p.value.value == 0.5  # value not updated
        p.min.value = 1.0
        assert p.value.value == p.min.value  # value was notified of min change

        p.max.value = 10.0
        assert p.nomax.value is False  # nomax was notified of max change
        p.nomax.value = True
        assert p.max.value == p.inf  # max was notified of nomax change

        p.min.value = -1.0
        p.value.value = 0.0
        p.max.value = 1.0
        assert p.value.value == 0.0  # value not updated
        p.max.value = -0.5
        assert p.value.value == p.max.value  # value was notified of max change

        p.min.value = -1.0
        p.value.value = 0.0
        p.max.value = 1.0
        p.min.value = p.max.value + 1.0
        assert p.min.value == -1.0  # min value rejected
        p.max.value = p.min.value - 1.0
        assert p.max.value == 1.0  # max value rejected

        p.min.value = -1.0
        p.value.value = -2.0
        assert p.value.value == p.min.value  # value within bounds
        p.max.value = 1.0
        p.value.value = 2.0
        assert p.value.value == p.max.value  # value within bounds


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
