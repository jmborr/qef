from __future__ import (absolute_import, division, print_function)

import numpy as np
from functools import wraps
from lmfit.model import Model

MIN_POS_DBL = np.nextafter(0, 1)  # minimum positive float


def prefix_params(param_expr):
    r"""Prepend parameter names with prefix in parameter expressions

    Parameters
    ----------
    param_expr: function
        bound method of a model returning an expression for a parameter in
        string format

    Returns
    -------
        function
    """
    @wraps(param_expr)
    def wrapper(model_instance):
        if not isinstance(model_instance, Model):
            raise TypeError('Function argument is not a Model instance')
        prefix = model_instance.prefix
        p_e = param_expr(model_instance)  # the parameter expression in str
        for prefixed_name in model_instance.param_names:
            name = prefixed_name.replace(prefix, '')  # drop the prefix
            p_e = p_e.replace(name, prefixed_name)
        return p_e
    return wrapper
