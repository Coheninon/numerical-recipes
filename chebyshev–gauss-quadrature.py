import numpy as np

def absc_and_weights(N):
    val = np.pi/N
    w_j = val
    x_j = []
    for j in range(1, N + 1):
        x_j += [np.cos(val*(j-0.5))]
    return x_j, w_j

def gau_che_int(func, absc, weights, N):
    g = lambda x: func(x) * ((1 - (x**2))**0.5)
    I = 0
    for j in range(N):
        I += weights * g(absc[j])
    return I

N = 2
f1 = lambda x: x**2
f2 = lambda x: (1+x)/((1-(x**2))**0.5)
f3 = lambda x: np.exp(x)
F1 = 2/3
F2 = np.pi
F3 = np.exp(1) - np.exp(-1)

for N in [10,100,1000]:
    x = absc_and_weights(N)[0]
    w = absc_and_weights(N)[1]
    d1 = abs(F1 - gau_che_int(f1, x, w, N))
    d2 = abs(F2 - gau_che_int(f2, x, w, N))
    d3 = abs(F3 - gau_che_int(f3,  x, w, N))
    print('Error for N=' + str(N) + ':\n' + 'in f1: ' + str(d1) \
    + ' '*(25 - len(str(d1))) + 'in f2: ' + str(d2) \
    + ' '*(25 - len(str(d2))) + 'in f3' + str(d3) + '\n')



