import numpy as np
import scipy

def f(x):
    y = 1/(6 + 4*np.cos(2*x))
    return y

xi = 0
xf = 0.5*np.pi

def trapn(N, xi, xf):
    sum = 0
    if N == 0:
        h = (xf - xi)/2
        sum = f(xi) + f(xf)
    else:
        h = (xf - xi)/(2**N)
        for i in range(1, 2**(N - 1) + 1):
            val = xi + (2*i - 1)*h
            sum += f(val)
    return h*sum

def trapdouble(xi, xf, N = 0, e = 0):
    k = trapn(N, xi, xf)
    if N == 0:
        e = k
        E = e
    else:
        E = abs(-0.5 * e + k)
        e = (0.5 * e) + k
    if E <= e * (2**(-53)):
        return e, N
    else:
        return trapdouble(xi, xf, N+1, e)

def pythontrap(xi, xf, N = trapdouble(xi, xf)[1]):
    h = (xf - xi)/(2 ** N)
    R = np.arange(xi, xf + h, h)
    f_R = f(R)
    return np.trapz(f_R, dx = h)

def analytic(x):
    return (1/(2*(5**0.5)) * np.arctan(np.tan(x)/(5**0.5)))

print('Value calculated by numeric function: ' + str(trapdouble(xi, xf)[0]) +' with ' + str(trapdouble(xi, xf)[1]) +\
      " steps\nValue calculated by python's built-in function: " + str(pythontrap(xi, xf, N = trapdouble(xi, xf)[1])) +\
      '\nValue calculated by analytic function: ' + str(analytic(xf)-analytic(xi)))
