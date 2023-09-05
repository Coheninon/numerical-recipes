import numpy as np
import matplotlib.pyplot as plt
c = np.zeros((400, 400), dtype=complex)
for i in range(c.shape[0]):
    if i == 0:
        re = -2.5
    else:
        re = c[0][i-1].real + 4/399

    for j in range(c.shape[0]):
        if j == 0:
            im = -1.5
        else:
            im = c[j-1][0].imag + 3/399
        c[j][i] = complex(re,im)
print(c)
z = np.zeros((400, 400), dtype=complex)
s = np.zeros((400, 400), dtype=int)
k = 0
while k < 25:
    z = z**2 + c
    s = s + (abs(z) <= 2)

    k += 1
print(s)
plt.imshow(s)
plt.show()