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

#Question 1

def hw7rho(x, y, z):
    return abs(x - 1)

def  hw7cylint (rho, rmax, xyzmin, xyzmax, n):
    x_i = np.random.uniform(xyzmin[0], xyzmax[0], n)
    y_i = np.random.uniform(xyzmin[1], xyzmax[1], n)
    z_i = np.random.uniform(xyzmin[2], xyzmax[2], n)
    V = (abs(xyzmax[0] - xyzmin[0])) * (abs(xyzmax[1] - xyzmin[1]))*(abs(xyzmax[2] - xyzmin[2]))
    
    p = hw7rho(x_i, y_i, z_i)
    a = np.array([rmax**2]*n)
    p_good = p[a >= ((x_i**2)+(y_i**2))]
    p_good_s = p_good**2
    n_good = len(p_good)
    
    dv = sum(p_good) / n
    dv_s = sum(p_good_s) / n
    m = V * dv
    dm = V * (((dv_s - (dv**2))/n)**0.5)

    #calculating xcm
    
    pxcm = x_i * hw7rho(x_i, y_i, z_i) / m
    pxcm_good = pxcm[a >= ((x_i**2)+(y_i**2))]
    pxcm_good_s = pxcm_good ** 2
    fxcm = sum(pxcm_good) / n
    fxcm_s = sum(pxcm_good_s) / n
    xcm = V * fxcm
    dxcm = V * (((fxcm_s - (fxcm**2))/n)**0.5)
    
    #calculating ycm

    pycm = y_i * hw7rho(x_i, y_i, z_i) / m
    pycm_good = pycm[a >= ((x_i**2)+(y_i**2))]
    pycm_good_s = pycm_good ** 2
    fycm = sum(pxcm_good) / n
    fycm_s = sum(pxcm_good_s) / n
    ycm = V * fycm
    dycm = V * (((fycm_s - (fycm**2))/n)**0.5)

    #calculating zcm
    
    pzcm = z_i * hw7rho(x_i, y_i, z_i) / m
    pzcm_good = pzcm[a >= ((x_i**2)+(y_i**2))]
    pzcm_good_s = pzcm_good ** 2
    fzcm = sum(pzcm_good) / n
    fzcm_s = sum(pzcm_good_s) / n
    zcm = V * fzcm
    dzcm = V * (((fzcm_s - (fzcm**2))/n)**0.5)
                   
    cm = (xcm, ycm, zcm)
    dcm = (dxcm, dycm, dzcm)

    return m, dm , cm, dcm, n_good

I = hw7cylint(hw7rho, 3, (-3,-3,-2), (3,3,2), 1000)
II = hw7cylint(hw7rho, 3, (-3,-3,-2), (3,3,2), 10000)
III = hw7cylint(hw7rho, 3, (-3,-3,-2), (3,3,2), 100000)
IV = hw7cylint(hw7rho, 3, (-3,-3,-2), (3,3,2), 1000000)

V = hw7cylint(hw7rho, 3, (1,-3,-1), (3,3,2), 1000000)

print('1.\nI. 1000 Guesses:\nm = ' + str(I[0]) + ' +- ' + str(I[1]) + '\Rcm = ' + str(I[2]) + ' +- ' +
      str(I[3]) + '\nNumber of points used: ' + str(I[4]))
print('\n\n2.\nII. 10000 Guesses:\nm = ' + str(II[0]) + ' +- ' + str(II[1]) + '\nRcm = ' + str(II[2]) + ' +- ' +
      str(II[3]) + '\nNumber of points used: ' + str(II[4])) 
print('\n\n3.\nIII. 100000 Guesses:\nm = ' + str(III[0]) + ' +- ' + str(III[1]) + '\nRcm = ' + str(III[2]) + ' +- ' +
      str(III[3]) + '\nNumber of points used: ' + str(III[4]))
print('\n\n4.\nIV. 1000000 Guesses:\nm = ' + str(IV[0]) + ' +- ' + str(IV[1]) + '\nRcm = ' + str(IV[2]) + ' +- ' +
      str(IV[3]) + '\nNumber of points used: ' + str(IV[4]))
print('\n\n5.\nV. 1000000 Guesses:\nm = ' + str(V[0]) + ' +- ' + str(V[1]) + '\nRcm = ' + str(V[2]) + ' +- ' +
      str(V[3]) + '\nNumber of points used: ' + str(V[4]))


#Question 2

def hw7mygaussj (A, b, s):
    A = np.array(A, dtype = float)
    b = np.array(b, dtype = float)
    Ainv = np.identity(len(A)).astype(float)
    for i in range(len(A)):
        if s:
            t = abs(A[i:,i])
            max_idx = t.argmax() + i

            max_row = A[max_idx,:].copy()
            copied_row = Ainv[max_idx,:].copy()
            copied_row_b = b[max_idx,:].copy()

            A[max_idx, :] = A[i, :]
            Ainv[max_idx, :] = Ainv[i, :]
            b[max_idx, :] = b[i, :]

            A[i, :] = max_row
            Ainv[i, :] = copied_row
            b[i, :] = copied_row_b

        val = A[i][i]
        A[i, :] = A[i, :]/val
        Ainv[i, :] = Ainv[i, :]/val
        b[i, :] = b[i, :]/val

        for j in range(len(A)):
            if i != j:
                val_2 = A[j][i]
                A[j,:] = A[j,:] - val_2 * A[i, :]
                Ainv[j, :] = Ainv[j, :] -  val_2 * Ainv[i, :]
                b[j, :] = b[j, :] -  val_2 * b[i,:]
    return Matrix(Ainv), Matrix(b)

A = Matrix([[12.113, 1.067, 9.574, 8.414, 0.098], [9.609 , 5.015, 8.814, 7.983, 7.692],
    [7.402, 0.081, 5.394, 0.417, 9.603], [1.451, 1.517, 3.741, 4.668, 2.601],
    [2.053, 1.576, 8.046, 8.152, 2.896]])
npA = np.array(A)
b = Matrix([[-0.046], [-11.655], [0.0623], [-1.351], [-0.227]])
npb = np.array(b)
I = Matrix([[1.,0.,0.,0.,0.],[0.,1.,0.,0.,0.],[0.,0.,1.,0.,0.],[0.,0.,0.,1.,0.],[0.,0.,0.,0.,1.]])



#Without Partial Pivoting

Ainv1, x1 = hw7mygaussj(A, b, False)

npx1 = np.array(x1)
npAinv1 = np.array(Ainv1)

Ax1 = Matrix(np.matmul(npA, npx1))

Ainvb1 = Matrix(np.matmul(npAinv1, npb))

AAinv1 = Matrix(np.matmul(npA, npAinv1))

#With Partial Pivoting

Ainv2, x2 = hw7mygaussj(A, b, True)

npx2 = np.array(x2)
npAinv2 = np.array(Ainv2)

Ax2 = Matrix(np.matmul(npA, npx2))

Ainvb2 = Matrix(np.matmul(npAinv2, npb))

AAinv2 = Matrix(np.matmul(npA, npAinv2))

#With Reff Function

A_reff = A.col_insert(5,b)
AI_reff = A.col_insert(5,I)

Ainv3 = AI_reff.rref()[0][:, 5:]
npAinv3 = np.array(Ainv3)

x3 = AI_reff.rref()[0][:, 5:]
npx3 = np.array(x3)

Ax3 = Matrix(np.matmul(npA, npx3))

Ainvb3 = Matrix(np.matmul(npAinv3, npb))

AAinv3 = Matrix(np.matmul(npA, npAinv3))

print('\n\nAinv:\n\nWithout Using Partial Pivoting:' + str(Ainv1) +
      '\n\nWith Partial Pivoting:' + str(Ainv2) + '\n\nUsing Reff Function:' + ' ' + str(Ainv3))

print('\n\nX:\n\nWithout Using Partial Pivoting:' + str(x1) +
      '\n\nWith Partial Pivoting:' + str(x2) + '\n\nUsing Reff Function:' + ' ' + str(x3))

print('\n\nA|x:\n\nWithout Using Partial Pivoting:' + str(Ax1) +
      '\n\nWith Partial Pivoting:' + str(Ax2) + '\n\nUsing Reff Function:' + ' ' + str(Ax3))

print('\n\nAinv|b:\n\nWithout Using Partial Pivoting:' + str(Ainvb1) +
      '\n\nWith Partial Pivoting:' + str(Ainvb2) + '\n\nUsing Reff Function:' + ' ' + str(Ainvb3))

print('\n\nA|Ainv:\n\nWithout Using Partial Pivoting:' + str(AAinv1) +
      '\n\nWith Partial Pivoting:' + str(AAinv2) + '\n\nUsing Reff Function:' + ' ' + str(AAinv3))




