import numpy as np
import matplotlib.pyplot as plt
from uncon_optimizer import uncon_optimizer
from scipy.optimize import minimize

def slanted_quadratic_function(x):
    """
    x: A 2xN matrix, where N is the number of column vectors.
    """
    beta = 1.5
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

def slanted_quadratic_ALL(x):
    return slanted_quadratic_function(x), slanted_quadratic_gradient(x)

def slanted_quadratic_single(x):
    """
    Wrapper function to accept a 1D array input.
    """
    x = np.reshape(x, (2, 1))
    return slanted_quadratic_function(x)[0]

def slanted_quadratic_gradient_single(x):
    """
    Wrapper function to accept and return 1D array for gradient.
    """
    x = np.reshape(x, (2, 1))
    return slanted_quadratic_gradient(x).flatten()


optimization_path = []
gradient_path = []

def callback(x):
    optimization_path.append(np.copy(x))
    gradient_path.append(np.max(np.abs(slanted_quadratic_gradient_single(x))))

result = minimize(fun=slanted_quadratic_single, x0=[-1,2], jac=slanted_quadratic_gradient_single, method='BFGS', callback=callback)

optimization_path = np.vstack(([-1,2],optimization_path))

Result = uncon_optimizer(slanted_quadratic_ALL, np.array([[-1],[2]]), 1E-6, options="SD OUTPUT-ALL")

#print(Result[0],Result[1],Result[3],Result[4])

plt.figure()
plt.plot(Result[4],"bo-",label = "Steepest Descent")
plt.plot(gradient_path,"go-", label = "BFGS (SciPy)")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Max Gradient")
plt.title("Slanted Quadratic Function - Steepest Descent")
plt.legend()
plt.grid()


x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-4, 4, 400)
X1, X2 = np.meshgrid(x1, x2)
x = np.vstack([X1.ravel(), X2.ravel()])  # Stack to make 2xN matrix

# Evaluate the function
Z = slanted_quadratic_function(x)
Z = Z.reshape(X1.shape)  # Reshape back to the grid shape

fig, axs = plt.subplots()
axs.contour(X1, X2, Z, 80, colors = 'k')
plot_xk = Result[3]
axs.plot(plot_xk[0,:],plot_xk[1,:],'b-',label = "Steepest Descent",zorder = 1)
axs.scatter(plot_xk[0,:],plot_xk[1,:],color = "red", zorder = 1)

axs.plot(optimization_path[:,0],optimization_path[:,1],'g-',label = "BFGS (SciPy)",zorder = 1)
axs.scatter(optimization_path[:,0],optimization_path[:,1],color = "red", zorder = 1)


plt.legend()

#---------------------Rosenbrock Function----------------------------------

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

def rosenbrock_ALL(x):
    return rosenbrock_function(x), rosenbrock_gradient(x)


def rosenbrock_single(x):
    """
    Wrapper function to accept a 1D array input.
    """
    x = np.reshape(x, (2, 1))
    return rosenbrock_function(x)[0]

def rosenbrock_gradient_single(x):
    """
    Wrapper function to accept and return 1D array for gradient.
    """
    x = np.reshape(x, (2, 1))
    return rosenbrock_gradient(x).flatten()


optimization_path = []
gradient_path = []

def callback(x):
    optimization_path.append(np.copy(x))
    gradient_path.append(np.max(np.abs(rosenbrock_gradient_single(x))))


initial_guess=[0, 0]
result = minimize(fun=rosenbrock_single, x0=initial_guess, method='BFGS',callback=callback)
#print(result)

optimization_path = np.vstack(([-3,1],optimization_path))


Result = uncon_optimizer(rosenbrock_ALL, np.array([[-3],[1]]), 1E-6, options="SD OUTPUT-ALL") 

#print(Result[0])
#print(Result[1])

x1 = np.linspace(-4, 4, 400)
x2 = np.linspace(-4, 4, 400)
X1, X2 = np.meshgrid(x1, x2)
x = np.vstack([X1.ravel(), X2.ravel()])  # Stack to make 2xN matrix

# Evaluate the function
Z = rosenbrock_function(x)
Z = Z.reshape(X1.shape)  # Reshape back to the grid shape

plt.figure()
plt.plot(Result[4],"bo-",label = "Steepest Descent")
plt.plot(gradient_path,"go-", label = "BFGS (SciPy)")

plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Max Gradient")
plt.title("Rosenbrock Function - Steepest Descent")
plt.legend()
plt.grid()

fig, axs = plt.subplots()
axs.contour(X1, X2, Z, 80, colors = 'k')
plot_xk = Result[3]
axs.plot(plot_xk[0,:],plot_xk[1,:],'b-',label = "Steepest Descent",zorder = 1)
axs.scatter(plot_xk[0,:],plot_xk[1,:],color = "red", zorder = 1)

axs.plot(optimization_path[:,0],optimization_path[:,1],'g-',label = "BFGS (SciPy)",zorder = 1)
axs.scatter(optimization_path[:,0],optimization_path[:,1],color = "red", zorder = 1)

plt.legend()
plt.show()

