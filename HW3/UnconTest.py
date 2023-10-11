import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from uncon_optimizer import uncon_optimizer

def bean_function(x):
    """
    x: A 2xN matrix, where N is the number of column vectors.
    """
    x1 = x[0, :]
    x2 = x[1, :]
    return (1 - x1)**2 + (1 - x2)**2 + 0.5 * (2 * x2 - x1**2)**2

def bean_function_gradient(x):
    """
    x: A 2xN matrix, where N is the number of column vectors.
    Returns: A 2xN matrix representing the gradient for each column vector.
    """
    x1 = x[0, :]
    x2 = x[1, :]

    df_dx1 = 2*x1**3+2*x1-4*x1*x2-2
    df_dx2 = 6*x2 - 2*x1**2 - 2

    return np.vstack([df_dx1, df_dx2])

def bean_function_ALL(x):
    return bean_function(x), bean_function_gradient(x)

x0 = np.array([[-3],[3]])
BFGS_Result = uncon_optimizer(bean_function_ALL,x0,1E-6, options = "BFGS OUTPUT-ALL")
SD_Result   = uncon_optimizer(bean_function_ALL,x0,1E-6, options = "SD OUTPUT-ALL")

plt.figure()
plt.plot(BFGS_Result[4],'go-',label = "BFGS")
plt.plot(SD_Result[4],'bo-', label = "Steepest Descent")
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Largest Gradient Norm')
plt.grid()
plt.legend()

x1 = np.linspace(-4, 4, 400)
x2 = np.linspace(-4, 4, 400)
X1, X2 = np.meshgrid(x1, x2)
x = np.vstack([X1.ravel(), X2.ravel()])  # Stack to make 2xN matrix

# Evaluate the function
Z = bean_function(x)
Z = Z.reshape(X1.shape)  # Reshape back to the grid shape

fig, axs = plt.subplots()
axs.contour(X1, X2, Z, 80, colors = 'k')
SD_plot_xk = SD_Result[3]
BFGS_plot_xk = BFGS_Result[3]
axs.plot(SD_plot_xk[0,:],SD_plot_xk[1,:],'b-',label = "SD Path",zorder = 1)
axs.scatter(SD_plot_xk[0,:],SD_plot_xk[1,:],color = "red", zorder = 1)

axs.plot(BFGS_plot_xk[0,:],BFGS_plot_xk[1,:],'g-',label = "BFGS Path",zorder = 1)
axs.scatter(BFGS_plot_xk[0,:],BFGS_plot_xk[1,:],color = "red", zorder = 1)
plt.legend()

plt.show()

