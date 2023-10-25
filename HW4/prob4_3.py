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


def plot_contour(mu, xlim, ylim, n_points=400):
    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)
    
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(n_points):
        for j in range(n_points):
            result = f_hat(np.array([[X[i, j]], [Y[i, j]]]), mu)
            Z[i,j] = result[0]

    plt.contourf(X, Y, Z, 50, cmap='viridis')
    plt.colorbar()
    plt.title(f"Contour plot for mu = {mu}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

mu = 0.5
#---------------------------Time to Optimize---------------------------------

diff = 10
rho  = 1.2
x    = np.array([[0],[0]])
k    = 0
while diff > 1E-4:
    x_past = x
    result = uncon_optimizer(f_hat, x, 1E-6, mu, options=None)

    x = result[0]
    mu = mu*1.2
    diff = np.abs(np.max(x_past) - np.max(x))
    print(diff)
    k = k + 1

print(x)