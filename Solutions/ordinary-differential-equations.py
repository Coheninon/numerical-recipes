import matplotlib.pyplot as plt
import os
import pickle
import sys
import numpy as np
import matplotlib as mpl
from scipy.interpolate import lagrange
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import CubicHermiteSpline
from scipy.special import jv
from scipy.misc import derivative
from numpy import sqrt, sin, cos, pi, abs, arctan,tan, linspace, polyfit
import random as rand
from sympy import *

def f(x):
    f1 = ((2*(x[1]**2))/x[0]) + x[0] - (x[0]**2)
    f0 = x[1]
    return np.array([f0, f1])

def euler(xt, f, h, start, stop):
    xt = np.array(xt)
    x = xt
    t = start
    xvals = []
    tvals = []
    while t < stop:
        xvals.append(x[0])
        tvals.append(t)
        t += h
        x = x + h*f(x)
    return xvals, tvals

def rk2(xt, f, h, start, stop):
    xt = np.array(xt)
    x = xt
    t = start
    xvals = []
    tvals = []
    while t < stop:
        xvals.append(x[0])
        tvals.append(t)
        t += h
        K1 = h * f(x)
        K2 = h * f(x + K1)
        x = x + 0.5*(K1 + K2)
    return xvals, tvals

def rk4(xt, f, h, start, stop):
    xt = np.array(xt)
    x = xt
    t = start
    xvals = []
    tvals = []
    while t < stop:
        xvals.append(x[0])
        tvals.append(t)
        t += h
        K1 = h * f(x)
        K2 = h * f(x + (K1/2))
        K3 = h * f(x + (K2 / 2))
        K4 = h * f(x + K3)
        x = x + (1/6) * (K1 + 2*K2 + 2*K3 +K4)
    return xvals, tvals

r = lambda x: 1/(1 + 0.5*np.cos(x))
t = linspace(0, 20*np.pi, int(((20*np.pi)//0.05)+1))
x_true = r(t)

xt = [2/3 ,0]
h = 0.05
x_euler, t_euler = euler(xt, f, h, 0, 20*np.pi)
x_rk_2, t_rk_2 = rk2(xt, f, h, 0, 20*np.pi)
x_rk_4, t_rk_4 = rk4(xt, f, h, 0, 20*np.pi)
dx_euler = x_euler - x_true
dx_rk_2 = x_rk_2 - x_true
dx_rk_4 = x_rk_4 - x_true
O2 = np.array(x_euler) * (h**2) /np.array(x_euler)
O3 = np.array(x_rk_2) * (h**3) / np.array(x_rk_2)
O5 = np.array(x_rk_4) * (h**5) / np.array(x_rk_4)

#print(x_euler)
#print(t_euler)
#Decided not to print because Python shrinks the results anyway

plt.plot(t_euler, x_euler)
plt.xlabel('ψ')
plt.ylabel('r')
plt.suptitle('Euler Method')
plt.show()

plt.plot(t_rk_2, x_rk_2)
plt.xlabel('ψ')
plt.ylabel('r')
plt.suptitle('Runge Kutta 2')
plt.show()

plt.plot(t_rk_4, x_rk_4)
plt.xlabel('ψ')
plt.ylabel('r')
plt.suptitle('Runge Kutta 4')
plt.show()

plt.plot(t, dx_euler, t, O2)
plt.xlabel('ψ')
plt.ylabel('dr')
plt.legend(['r_euler - r_real','h^2'], loc='best')
plt.suptitle('Error in Euler Method')
plt.show()

plt.plot(t, dx_rk_2, t, O3)
plt.xlabel('ψ')
plt.ylabel('dr')
plt.legend(['r_runge_kutta2 - r_real','h^3'], loc='best')
plt.suptitle('Error in Runge Kutta 2')
plt.show()

plt.plot(t, dx_rk_4, t, O5)
plt.xlabel('ψ')
plt.ylabel('dr')
plt.legend(['r_runge_kutta4 - r_real','h^5'], loc='best')
plt.suptitle('Error in Runge Kutta 4')
plt.show()

