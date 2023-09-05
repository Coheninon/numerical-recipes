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

def Zrika(n, xa, xb, yi, yf, f):
    ntot = 0
    ngood = 0
    dots = np.array([])
    while (ngood < n):
        x = np.random.uniform(xa, xb, n - ngood)
        y = np.random.uniform(yi, yf, n - ngood)
        dots = np.hstack((dots, x[y <= f(x)]))
        ntot += (n - ngood)
        ngood = len(dots)
    return (dots,ngood/ntot)
    

def Inverse(f, n, g):
    ntot = 0
    ngood = 0
    dots = np.array([])
    while (ngood < n):
        u = np.random.uniform(0,1, n - ngood)
        y = g(u)
        dots = np.hstack((dots, y))
        ntot += (n - ngood)
        ngood = len(dots)
    return (dots, ngood/ntot)


f = lambda x: 1/(1 + x**2)
g = lambda x: np.tan((2*x - 1) * np.arctan(6))
p = lambda x: (1/(1 + x**2)) * (np.cos(x)**2)
F = lambda x: (1/(1 + x**2)) * (1/(2 * np.arctan(6)))
P = lambda x: (1/(1 + x**2)) * (np.cos(x)**2) * (1/(2 * np.arctan(6)))
                                                    
n = 100000

hist1, eff1 = Zrika (n, -6, 6, -0.027, 1, f)
                                                    
hist2, eff2 = Inverse (f, n, g)

                                                                                               
plt.hist(hist1, bins = 50)
plt.suptitle('Zrika Plot')
plt.show()

plt.hist(hist2, bins = 50)
plt.suptitle('Inverse Plot')
plt.show()

print('The Zrika method produced an effiecency of: ' + str(eff1) + '\nThe Inverse method produced an effiecency of: ' + str(eff2))
#As expected, the efficiency of Zrika method is lower because there is a lot of empty space being covered by the rectangle.
#Furthemore, in the Inverse method I used the analytic solution, therfore I didnt use any unnecessary points.

def Comb(n, g, F, P):
    ntot = 0
    ngood = 0
    dots = np.array([])
    while (ngood < n):
        u = np.random.uniform(0, 1, n - ngood)
        t = np.random.uniform(0, 1, n - ngood)
        x = g(t)
        dots = np.hstack((dots, x[u*F(x) <= P(x)]))
        ntot += (n - ngood)
        ngood = len(dots)
    return (dots, ngood/ntot) 

hist3, eff3 = Comb (n, g, F, P)
                                                    
                                                                                               
plt.hist(hist3, bins=50)
x = linspace(-6, 6, 10000)
plt.plot(x, f(x)*14500, x, p(x)*14500)
plt.legend(['f(x)', 'g(x)'], loc='best')
plt.suptitle('Comb Plot')
plt.show()



print('The Combined method produced an effiecency of: ' + str(eff3))
        
                                                    
