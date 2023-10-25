import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from uncon_optimizer import uncon_optimizer
#------------------------Function Definition---------------------------------

x1, x2, mu = sp.symbols('x1, x2, mu')


term = sp.Max(0, (1/4) * x1**2 + x2**2 - 1)
f_hat = x1 + 2 * x2 + (mu / 2) * term**2

f_hat_x1_sym = sp.diff(f_hat,x1)
f_hat_x2_sym = sp.diff(f_hat,x2)

f_hat_x1_Lambda = sp.lambdify((x1,x2,mu),f_hat_x1_sym,'numpy')
f_hat_x2_Lambda = sp.lambdify((x1,x2,mu),f_hat_x2_sym,'numpy')

def f_hat_grad (x1,x2,mu):
    f_hat_x1 = f_hat_x1_Lambda(x1,x2,mu)
    f_hat_x2 = f_hat_x2_Lambda(x1,x2,mu)

    return np.vstack([f_hat_x1,f_hat_x2])

def f_hat(x,mu):
    grad = f_hat_grad(x[0,0],x[1,0],mu)

    term = max(0, (1/4) * x[0,0]**2 + x[1,0]**2 - 1)
    result = x[0,0] + 2 * x[1,0] + (mu/ 2) * term**2
    return result, grad 


#---------------------------Time to Optimize---------------------------------
result = uncon_optimizer(f_hat, np.array([[0],[0]]), 1E-6, mu, options=None)
print(result)