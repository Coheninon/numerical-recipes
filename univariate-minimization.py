import numpy as np
import matplotlib.pyplot as plt

def find_min(a, c, b, e, x0 = 0):
    f = lambda x: -np.sin(4 * x ** 2)
    x = c - 0.5*(((f(c)-f(b)) * (c-a)**2) - ((f(c)-f(a)) * (c-b)**2))/(((f(c)-f(b)) * (c-a)) - ((f(c)-f(a)) * (c-b)))    
    if (x > b) and (x < c):
        if (f(x) > f(b)) and (f(x) < f(c)):
            new_a = a
            new_b = b
            new_c = x
        elif (f(x) < f(b)) and (f(x) < f(c)):
            new_a = b   
            new_b = x
            new_c = c
    elif (x < b) and (x > a):
        if (f(x) > f(b)) and (f(x) < f(a)) :
            new_a = x
            new_b = b
            new_c = c
        elif (f(x) < f(b)) and (f(x) < f(a)):
            new_a = a
            new_b = x
            new_c = b
     
    if abs(x - x0) <= e:
        return (new_a, new_c, new_b, x, f(x))
    return find_min(new_a, new_c, new_b, e, x)

a = 0.1
b = 1
c = (a+b)/2
a1, c1, b1, x1, fx1 = find_min(a, c, b, 0.1)
a2, c2 , b2, x2, fx2 = find_min(a, c, b, 0.00001)
real_x_min = (np.pi / 8)**0.5
real_fx_min = -1
errx1 = abs(real_x_min - x1)
errfx1 = abs(real_fx_min - fx1)
errx2 = abs(real_x_min - x2)
errfx2 = abs(real_fx_min - fx2)

print('\n\nThe analytic solution is: [' + str(real_x_min) + ', ' + str(real_fx_min) + ']\n\nFor e = 10^-1, the numeric solution is:'
      + ' [' + str(x1) + ', ' + str(fx1) + '] \nAnd the error is: [' + str(errx1) + ', ' + str(errfx1) + ']' +
      '\n\nFor e = 10^-5, the numeric solution is:'
      + ' [' + str(x2) + ', ' + str(fx2) + '], \nAnd the error is: [' + str(errx2) + ', ' + str(errfx2) + ']')
 
