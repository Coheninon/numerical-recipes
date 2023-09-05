import matplotlib.pyplot as plt
import os
import pickle
import sys
import numpy as np
import matplotlib as mpl
import scipy
from scipy.interpolate import lagrange
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import CubicHermiteSpline
from scipy.special import jv
from scipy.misc import derivative
from numpy import sqrt, sin, cos, pi, abs, arctan,tan, linspace, polyfit
import random as rand
from sympy import *
import numdifftools as nd

def hw10GDstep(f, x0):
    val = x0
    minvals = [val]
    fvals = [f(val)]
    iter_count = 0
    valx = [x0[0]]
    valy = [x0[1]]
    norm = np.sqrt(val[0]**2 + val[1]**2)
    while iter_count < 1000 and norm > 10**-4:
        val = val - 0.2*nd.Gradient(f)(val)
        minvals.append(val)
        valx.append(val[0])
        valy.append(val[1])
        fvals.append(f(val))
        norm = np.sqrt(val[0]**2 + val[1]**2)
        iter_count += 1
    return minvals, fvals, iter_count, valx, valy

def f(x):
    return np.cos(x[0]) + np.sin(x[1]) + 0.2*x[0]**2 + 0.25*x[1]**2


minvals1, fvals1, iters1, valx1, valy1 = hw10GDstep(f, [7, 7])
minvals2, fvals2, iters2, valx2, valy2 = hw10GDstep(f, [-7, -7])
minvals3, fvals3, iters3, valx3, valy3 = hw10GDstep(f, [0, -7])

min1 = np.array(minvals1)
min2 = np.array(minvals2)
min3 = np.array(minvals3)


x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)

X, Y = np.meshgrid(x, y)
Z = f([X, Y])
plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.plot(valx1, valy1, '.')
plt.plot(valx2, valy2, '.')
plt.plot(valx3, valy3, '.')
plt.axis('equal')
plt.show()




