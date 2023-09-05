import numpy as np
import matplotlib.pyplot as plt

#Question 1

#Creating empty matrices\columns for the values to fill

mat = np.zeros((11,4), dtype = float)
x_n = np.zeros((11,), dtype = float)
r_n = np.zeros((11,), dtype = float)
p_n = np.zeros((11,), dtype = float)
q_n = np.zeros((11,), dtype = float)

#Setting the values of each set

for i in range (11):
    if i == 0:
        r_n[i] = 0.99996
        p_n[i] = 1
        q_n[i] = 1
        x_n[i] = 1/(3**i)
    elif i == 1:
        r_n[i] = r_n[i-1]/3
        p_n[i] = 0.33332
        q_n[i] = 0.33332
        x_n[i] = 1 / (3 ** i)
    else:
        r_n[i] = r_n[i-1]/3
        p_n[i] = (p_n[i-1] * (4/3)) - (p_n[i-2] * (1/3))
        q_n[i] = (q_n[i - 1] * (10 / 3)) - q_n[i - 2]
        x_n[i] = 1 / (3 ** i)

#Building the martrix "mat"

mat[0:,0] = x_n
mat[0:,1] = r_n
mat[0:,2] = p_n
mat[0:,3] = q_n

#Creating the graphs

lst = ['x[n]', 'r[n]', 'p[n]', 'q[n]']
for i in range(4):
    plt.plot(range(11), mat[0:, i])
    plt.xlabel('n')
    plt.ylabel(lst[i])
    plt.show()

#Question 2

#Creating empty matrices\columns for the values to fill

mat_2 = np.zeros((11,3), dtype = float)
mat_2[0:,0] = mat[0:,0] - mat[0:,1]
mat_2[0:,1] = mat[0:,0] - mat[0:,2]
mat_2[0:,2] = mat[0:,0] - mat[0:,3]

#Creating the graphs

lst2 = ['x[n]-r[n]', 'x[n]-p[n]', 'x[n]-q[n]']
for i in range(3):
    plt.plot(range(11), mat_2[0:, i])
    plt.xlabel('n')
    plt.ylabel(lst2[i])
    plt.show()

#Analyzing the errors

Error_mat = np.zeros((11,3))
Error_mat[0:,0] = mat_2[0:,0]
Error_mat[0:,1] = mat_2[0:,1]
Error_mat[0:,2] = mat_2[0:,2]
print(Error_mat)

#x_n - r_n is converging to a constant error
#x_n - p_n is exponentially decreasing 
#x_n - q_n is exponentially increasing
