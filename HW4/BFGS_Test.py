import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from uncon_optimizer import Uncon_BFGS

def f(x):
    '''
    FUNCTION TO BE OPTIMISED
    '''
    d = len(x)
    return sum(100*(x[i+1]-x[i]**2)**2 + (x[i]-1)**2 for i in range(d-1))

def slanted_quadratic_function(x):
    """
    x: A 2xN matrix, where N is the number of column vectors.
    """
    beta = 1.9
    x1 = x[0, :]
    x2 = x[1, :]
    return x1**2 + x2**2 - beta*x1*x2

def slanted_quadratic_gradient(x):
    """
    x: A 2xN matrix, where N is the number of column vectors.
    Returns: A 2xN matrix representing the gradient for each column vector.
    """
    x1 = x[0, :]
    x2 = x[1, :]

    df_dx1 = 2*x1 - 1.5*x2
    df_dx2 = 2*x2 - 1.5*x1

    return np.vstack([df_dx1, df_dx2])

def slanted_quadratic_ALL(x,mu):
    return slanted_quadratic_function(x), slanted_quadratic_gradient(x)


def rosenbrock_function(x):
    """
    x: A 2xN matrix, where N is the number of column vectors.
    """
    x1 = x[0, :]
    x2 = x[1, :]
    return (1 - x1)**2 + 100*(x2 - x1**2)**2

def rosenbrock_gradient(x):
    """
    x: A 2xN matrix, where N is the number of column vectors.
    Returns: A 2xN matrix representing the gradient for each column vector.
    """
    x1 = x[0, :]
    x2 = x[1, :]

    df_dx1 = (-2*(1-x1)-400*x1*(x2 - x1**2))
    df_dx2 = 200*(x2-x1**2)

    return np.vstack([df_dx1, df_dx2])

def rosenbrock_ALL(x, mu):
    return rosenbrock_function(x), rosenbrock_gradient(x)



x0 = np.array([[1.2],[1.2]])
mu = 0;


result = Uncon_BFGS(rosenbrock_ALL, x0, 1E-6, mu)
print (result)