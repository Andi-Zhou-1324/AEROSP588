import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from uncon_optimizer import uncon_optimizer
from uncon_optimizer import Uncon_BFGS

"""
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

    term = np.maximum(0, 0.25 * x[0,:]**2 + x[1,:]**2 - 1)    
    result = x[0,:] + 2 * x[1,:] + (mu/ 2) * term**2
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

mu = 0.5
#---------------------------Time to Optimize---------------------------------

def f_plot(x):
    return x[0,:] + 2 * x[1,:]

def f_hat_plot(x,mu):
    term = 0.25 * x[0,:]**2 + x[1,:]**2 - 1

    term[term<0] = 0    

    result = x[0,:] + 2 * x[1,:] + (mu/ 2) * term**2
    return result 

def plot_contour(x_vec, mu):
    x = np.linspace(-3, 3, 400) # You can adjust these bounds as needed
    y = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, y)
    xy = np.vstack((X.ravel(), Y.ravel()))

    Z = f_hat_plot(xy, mu)
    Z_f = f_plot(xy)

    Z = Z.reshape(X.shape)
    Z_f = Z_f.reshape(X.shape)

    Z_constraint = (1/4) * X**2 + Y**2 - 1


    plt.figure(figsize=(8,8))
    contour = plt.contour(X, Y, Z, 100, colors = 'black', alpha = 0.5)
    plt.contourf(X, Y, Z_f, 10, cmap='viridis')
    plt.contour(X, Y, Z_constraint, 100, levels = [0], colors = 'red')
    plt.plot(x_vec[0,:],x_vec[1,:], color = 'cyan',label = 'Optimization Path')
    plt.scatter(x_vec[0,:],x_vec[1,:], color = 'cyan')

    plt.title(f"mu = {mu}")
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    

diff = 10
rho  = 1.2
x    = np.array([[0],[0]])
k    = 0
x_vec = np.array(x)

while diff > 1E-4:
    x_past = x
    result = uncon_optimizer(f_hat, x, 1E-6, mu, options = None)

    x = result[0]
    x_vec = np.hstack((x_vec,x))
    mu = mu*rho
    diff = np.abs(np.max(x_past) - np.max(x))
    print(diff, k)
    k = k + 1
    if k == 1 or k == 10 or k == 20:
        plot_contour(x_vec, mu)
plot_contour(x_vec, mu)



print(x)
print(mu)


"""
#------------------------Interior Penalty------------------------------------
x1, x2, mu = sp.symbols('x1, x2, mu')


term = mu*sp.log(-(1/4)*x1**2 - x2**2 + 1)
f_hat_sym = x1 + 2 * x2 - term

f_hat_x1_sym = sp.diff(f_hat_sym,x1)
f_hat_x2_sym = sp.diff(f_hat_sym,x2)

f_hat_x1_Lambda = sp.lambdify((x1,x2,mu),f_hat_x1_sym,'numpy')
f_hat_x2_Lambda = sp.lambdify((x1,x2,mu),f_hat_x2_sym,'numpy')

def f_hat_grad (x1,x2,mu):
    f_hat_x1 = f_hat_x1_Lambda(x1,x2,mu)
    f_hat_x2 = f_hat_x2_Lambda(x1,x2,mu)

    return np.vstack([f_hat_x1,f_hat_x2])

def f_hat(x,mu):
    grad = f_hat_grad(x[0,0],x[1,0],mu)

    term =  mu*np.log(-(1/4)*x[0]**2 - x[1]**2 + 1)
    f_hat = x[0] + 2 * x[1] - term
    return f_hat, grad 


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

def f_plot(x):
    return x[0,:] + 2 * x[1,:]

def f_hat_plot(x,mu):
    term =  mu*np.log(-(1/4)*x[0,:]**2 - x[1,:]**2 + 1)

    result = x[0,:] + 2 * x[1,:] - term    
    return result 

def plot_contour(x_vec, mu):
    x = np.linspace(-3, 3, 400) # You can adjust these bounds as needed
    y = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, y)
    xy = np.vstack((X.ravel(), Y.ravel()))

    Z = f_hat_plot(xy, mu)
    Z_f = f_plot(xy)

    Z = Z.reshape(X.shape)
    Z_f = Z_f.reshape(X.shape)

    Z_constraint = (1/4) * X**2 + Y**2 - 1


    plt.figure(figsize=(8,8))
    contour = plt.contour(X, Y, Z, 100, colors = 'black', alpha = 0.5)
    plt.contourf(X, Y, Z_f, 10, cmap='viridis')
    plt.contour(X, Y, Z_constraint, 100, levels = [0], colors = 'red')
    plt.plot(x_vec[0,:],x_vec[1,:], color = 'cyan',label = 'Optimization Path')
    plt.scatter(x_vec[0,:],x_vec[1,:], color = 'cyan')

    plt.title(f"mu = {mu}")
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.grid(True)
    plt.axis('equal')
    plt.show()



    
x    = np.array([[1],[0]])

#plot_contour(mu, xlim = (-3,3), ylim = (2,2), n_points=400)


mu = 1

diff = 10
rho  = 0.8
k    = 0
x_vec = np.array(x)

while diff > 1E-4:
    x_past = x
    result = uncon_optimizer(f_hat, x, 1E-6, mu, options = None)

    x = result[0]
    x_vec = np.hstack((x_vec,x))

    mu = mu*rho
    diff = np.abs(np.max(x_past) - np.max(x))
    #print(diff, k)
    k = k + 1
    #if k == 1 or k == 10 or k == 20:
        #plot_contour(x_vec, mu)

#plot_contour(x_vec, mu)
print(x)
print(mu)
