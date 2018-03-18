import os
import sys
import importlib
import inspect
from lmfit.models import Model

# Import models from files in this directory
this_dir = os.path.dirname(locals()['__file__'])
sys.path.insert(0, this_dir)
try:
    for filename in os.listdir(this_dir):
        module_name, extension = os.path.splitext(filename)
        if extension == '.py':
            m = importlib.import_module(module_name)
            for name, a_type in inspect.getmembers(m):
                if hasattr(a_type, '__bases__') and Model in a_type.__bases__:
                    globals()[name] = a_type
finally:
    sys.path.remove(this_dir)
