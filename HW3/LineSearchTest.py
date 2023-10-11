import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from LineSearch import LineSearch_BackTrack,LineSearch_Bracketing,phi,d_phi
from LineSearch_Bracket import LineSearch_Bracketing_new

def f(x):
    x1 = x[0,:]
    x2 = x[1,:]
    return 0.1*x1**6 - 1.5*x1**4 + 5*x1**2 + 0.1*x2**4 + 3*x2**2 - 9*x2 + 0.5*x1*x2

def d_f(x):
    x1 = x[0,:]
    x2 = x[1,:]
    df_dx1 = 0.6*x1**5 - 6*x1**3 + 10*x1 + 0.5*x2
    df_dx2 = 0.4*x2**3 + 6*x2 - 9 + 0.5*x1
    return np.array([[df_dx1], [df_dx2]])  # Constructing a column vector

def func(x):
    return f(x), d_f(x)

x_k = np.array([[-1.25],[1.25]])
p_k = np.array([[4],[0.75]])

plt.figure()
plt.plot(np.linspace(0,1.2,1000),f(x_k + np.linspace(0,1.2,1000)*p_k),label = "Function Evaluation")

a_init = 1.2

mu_1 = 1E-4
mu_2 = 0.9
sigma = 2

result = LineSearch_Bracketing_new(a_init, func,mu_1,mu_2,sigma,x_k,p_k)


print(result[0], result[1])


#plt.axhline(func(x_k),color = "red",label = "Line of Sufficient Decrease")
plt.scatter(result[2],result[3],label = "Bracketing Points",facecolor = 'orange')
plt.scatter(result[4],result[5],label = "Pinpoint Points",facecolor = 'grey')

plt.xlabel('a')
plt.ylabel('f')
plt.legend()
plt.show()
