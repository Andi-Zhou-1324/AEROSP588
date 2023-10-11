import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from uncon_optimizer import uncon_optimizer
#---------------------------------------------------------

def rosenbrock_N(x):
    """Evaluate the Rosenbrock function."""
    x = x.flatten()  # Convert column array to 1D for easier indexing
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosenbrock_N_gradient(x):
    """Evaluate the gradient of the Rosenbrock function."""
    x = x.flatten()  # Convert column array to 1D for easier indexing
    n = len(x)
    grad = np.zeros(n)
    
    # For the first variable
    grad[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    
    # For the middle variables
    for j in range(1, n-1):
        grad[j] = 200*(x[j] - x[j-1]**2) - 400*x[j]*(x[j+1] - x[j]**2) - 2*(1-x[j])
    
    # For the last variable
    grad[-1] = 200*(x[-1]-x[-2]**2)
    
    return grad.reshape(-1, 1)  # Return as a column array

def rosenbrock_N_ALL(x):
    return rosenbrock_N(x), rosenbrock_N_gradient(x)



#print(Iter_count)


#-------------SCIPY SECTION------------------------
import numpy as np
from scipy.optimize import minimize

def rosenbrock(x):
 #  Evaluate the Rosenbrock function.
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosenbrock_grad(x):
 #   Evaluate the gradient of the Rosenbrock function.
    n = len(x)
    grad = np.zeros(n)
    
    # For the first variable
    grad[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    
    # For the middle variables
    for j in range(1, n-1):
        grad[j] = 200*(x[j] - x[j-1]**2) - 400*x[j]*(x[j+1] - x[j]**2) - 2*(1-x[j])
    
    # For the last variable
    grad[-1] = 200*(x[-1]-x[-2]**2)
    
    return grad

N_vec = np.array([2,4,8,16,32,64])

# Test the optimization
SciPy_Iter_Count = np.array([])
for i in N_vec:
    initial_guess = np.zeros(i)+2  # Starting from a random point in 3D
    result = minimize(fun=rosenbrock, x0=initial_guess, jac=rosenbrock_grad, method='BFGS')
    SciPy_Iter_Count = np.append(SciPy_Iter_Count, result.nit)


Iter_count = np.array([])

for i in N_vec:
    x0 = np.zeros([i,1])+2
    Result = uncon_optimizer(rosenbrock_N_ALL,x0,1E-6,"SD OUTPUT-ALL")
    Iter_count = np.append(Iter_count,len(Result[4]))



plt.figure()
plt.plot(N_vec, Iter_count, '-bo',label = "Steepest Descent")
plt.plot(N_vec, SciPy_Iter_Count, '-go', label = "BFGS (SciPy)")
plt.legend()
plt.xlabel("Dimensions (N)")
plt.ylabel("Iteration Count")
plt.grid()

"""
plt.figure()
plt.plot(Result[-1],'-b')
plt.xlabel("Iteration")
plt.ylabel("Max Absolute Gradient")
plt.yscale("log")
plt.grid()
"""


plt.show()