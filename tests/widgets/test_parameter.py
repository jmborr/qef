#!/usr/bin/env python
# -*- coding: utf-8 -*-\

from __future__ import (absolute_import, division, print_function)

import os
import pytest
import lmfit
from qef.widgets.parameter import ParameterWithTraits


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
        assert p.t_val == 42 and p.vary == True
        p.set(expr='hello')
        assert p.expr == 'hello' and p.t_expr == 'hello'
        assert p.vary is False and p.tvary is False
        p.set(value=42)
        assert p.t_val == 42 and p.vary == False
        assert p._expr is None and p.t_expr is None


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
