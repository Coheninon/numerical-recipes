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

def hw8lu(A):
    sizeA = len(A)
    U = np.zeros((sizeA, sizeA))
    L = np.zeros((sizeA, sizeA))
    for j in range(len(A)):
        for i in range(j+1):
            u = np.matmul(L[i,:i],U[:i,j])
            U[i, j] = A[i, j] - u
        for i in range(j, len(A)):
            l = np.matmul(L[i,:j],U[:j,j])
            L[i, j] = (1/U[j,j])*(A[i,j] - l)
    return L, U

def hw8LUorSVD(A,b):
    svals = scipy.linalg.svdvals(A)
    sing = ((min(svals)/max(svals)) < 10**(-5))
    if sing:
        TYPE = 'SVD'
        svals = scipy.linalg.svdvals(A)
        [U, S, V] = np.linalg.svd(A)
        Snew = (S > 10 ** -5) * S
        Sinv = np.concatenate((1 / Snew[Snew > 0], Snew[Snew <= 0]))
        n = len(Sinv)
        Ainv = V.T.dot((Sinv * np.eye(n)).dot(U.T))
        x = Ainv.dot(b)
    else:
        TYPE = 'LU'
        y = np.zeros((len(A), 1))
        yinv = np.zeros((len(A),len(A)))
        x = np.zeros((len(A), 1))
        Ainv = np.zeros((len(A), len(A)))
        I = np.identity(len(A))
        L, U = hw8lu(A)
        for i in range(len(A)):
            y_sum = np.matmul(L[i, :i], y[:i, 0])
            y[i, 0] = (1/L[i,i])*(b[i] - y_sum)
            for j in range(len(A)):
                yinv_sum = np.matmul(L[i, :i], yinv[:i, j])
                yinv[i,j] = (1/L[i,i])*(I[i,j] - yinv_sum)
        for i in range(len(A)-1, -1,-1):
            x_sum = np.matmul(U[i, i:],x[i:, 0])
            x[i, 0] = (1/U[i,i])*(y[i] - x_sum)
            for j in range(len(A)-1, -1, -1):
                Ainv_sum = np.matmul(U[i,i:],Ainv[i:,j])
                Ainv[i,j] = (1/U[i,i])*(yinv[i,j] - Ainv_sum)
    return x, Ainv, TYPE

A1 = np.array([[1,2,3],[4,5.5,6],[7,8,9]])
A2 = np.array([[1,2,3],[2,4,6],[7,8,9]])
A3 = np.array([[1,2,3],[4,5,6],[5,7,9]])
b = np.array([[1],[3],[5]])
x1, Ainv1, TYPE1 = hw8LUorSVD(A1,b)
x2, Ainv2, TYPE2 = hw8LUorSVD(A2,b)
x3, Ainv3, TYPE3 = hw8LUorSVD(A3,b)

    
print('\n\n\nThe method used for the first matrix was "' + TYPE1 + '".\n\nx:\n"' +
str(x1) + '\n\nAinv:\n' + str(Ainv1) + '\n\nA*x:\n' + str(np.matmul(A1,x1)))

    
print('\n\n\nThe method used for the second matrix was "' + TYPE2 + '".\n\nx:\n"' +
str(x2) + '\n\nAinv:\n' + str(Ainv2) + '\n\nA*x:\n' + str(np.matmul(A2,x2)))

    
print('\n\n\nThe method used for the third matrix was "' + TYPE3 + '".\n\nx:\n"' +
str(x3) + '\n\nAinv:\n' + str(Ainv3) + '\n\nA*x:\n' + str(np.matmul(A3,x3)))

    
