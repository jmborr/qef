import numpy as np
from scipy.interpolate import interp1d
from lmfit import Model
from matplotlib import pyplot as plt
""" fitting the tabulated Model to some arbitrary points"""


class fitting ():

    def __init__(self,x,y,a,r):
        self.x=x
        self.y=y
        self.a=a
        self.r=r

    def intrpl (self):

        return self.a*interp1d(self.x, self.y, fill_value='extrapolate', kind='cubic')(self.x-self.r)

    def model (self):
        mod = Model(self.intrpl)
        mod.set_param_hint('r', value=self.r, min=0.0, max=1.0)
        mod.set_param_hint('a', value=self.a, min=0.0)
        mod.set_param_hint('y', value=self.y, vary=False)

        pars = mod.make_params()

        R=mod.eval(pars, x=self.x)

        return (R)

