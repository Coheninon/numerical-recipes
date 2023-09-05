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

#Question 1

f = lambda x: x * sin(2*x) * abs(x - (3/4))
x = linspace(0, 1.5, 11)
xnew = linspace(0, 1.5, 500)
poly1 = polyfit(x, f(x), 8)
P1 = np.poly1d(poly1)
cubic = CubicSpline(x, f(x))
plt.plot(x, f(x),'.',xnew, f(xnew), xnew, P1(xnew), xnew, cubic(xnew))
plt.legend(['data','f(x)','poly','cubic'], loc='best')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.suptitle('Question 1')
plt.show()

#Question 2

x = lambda t: np.cos(t)
y = lambda t: np.sin(t)
t = np.arange(0, 2.01*np.pi , 0.01*np.pi)
t1 = np.arange(0, 2.5*np.pi , 0.5*np.pi)
n = np.linspace(0, 2*np.pi, 100)

plt.axes().set_aspect("equal")

cubicx = interp1d(t1, x(t1), kind='cubic')
cubicy = interp1d(t1, y(t1), kind='cubic')


lagx = lagrange(t1, x(t1))
lagy = lagrange(t1, y(t1))

cubicx2 = CubicSpline(t1, x(t1), bc_type = 'periodic')
cubicy2 = CubicSpline(t1, y(t1), bc_type = 'periodic')

plt.plot(x(t1), y(t1), '.', x(n), y(n), cubicx(n), cubicy(n), lagx(n), lagy(n), cubicx2(n), cubicy2(n))
plt.legend(['data','f(x)','Cubic','Lagrange','Updated cubic'], loc='best')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.suptitle('Question 2')
plt.show()

#We must include 2pi in order for the circle to be closed (we want to use all 5 points for the interpolation). If we are doing interpolation in the interval of [0,2pi), the circle will not be full.

#Question 3

x = [-3,-2,-1,0,1,2,3]
y = [-1,-1,-1,0,1,1,1]
f = interp1d(x,y)
n = linspace(-3,3,1000)
der = [0, 0, 0, 1, 0, 0, 0]
cubic = CubicSpline(x, y, bc_type = 'natural')
hermite = CubicHermiteSpline(x, y, dydx = der)
plt.plot(x, y, '.', n, f(n), n, cubic(n), n, hermite(n))
plt.legend(['data','f(x)', 'cubic','hermite'], loc='best')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.suptitle('Question 3 i.')
plt.show()

x = np.arange(0,26,1)
y = jv(1,x)
der = np.zeros(26,)
der[1:] = np.diff(y)/np.diff(x)
f = interp1d(x,y)
n = linspace(0,25,1000)
cubic = CubicSpline(x, y, bc_type='natural')
hermite = CubicHermiteSpline(x, y, dydx = der)
plt.plot(x, y, '.', n, f(n), n, cubic(n), n, hermite(n))
plt.legend(['data','f(x)', 'Cubic','Hermite'], loc='best')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.suptitle('Question 3 ii.')
plt.show()


