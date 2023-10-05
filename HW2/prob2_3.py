import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from LineSearch import LineSearch_BackTrack,LineSearch_Bracketing,phi,d_phi,Pinpoint

##Importing libraries

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

alpha_0 = 0.05
mu_1 = 1E-4
rho  = 0.7
x = np.array([[-1.25],[1.25]])
p_k = np.array([[4],[0.75]])

alpha, f_x, alpha_vec, phi_alpha_vec = LineSearch_BackTrack(alpha_0, mu_1, rho,f,d_f,x,p_k)

plt.figure()
plt.plot(np.linspace(0,1.2,1000),f(x + np.linspace(0,1.2,1000)*p_k),label = "Function Evaluation")
plt.axhline(f(x),color = "red",label = "Line of Sufficient Decrease")
plt.scatter(alpha_vec,phi_alpha_vec,label = "Back Tracking Points",facecolor = 'orange')
plt.xlabel('a')
plt.ylabel('f')
plt.legend()


#-------------------------------------Bracketing Line Search----------------------------------------------

sigma = 2
mu_2 = 0.9
a_init = 1.2
x_k = x

phi_0 = phi(f,x_k, 0,p_k)
d_phi_0 = d_phi(d_f, x_k, 0, p_k)

alpha, f_x = LineSearch_Bracketing (a_init, phi_0, d_phi_0, mu_1, mu_2, sigma, phi, d_phi,x_k,p_k,d_f,f)

print(alpha)
print(f_x)

plt.figure()
plt.plot(np.linspace(0,1.2,1000),f(x + np.linspace(0,1.2,1000)*p_k),label = "Function Evaluation")
plt.axhline(f(x),color = "red",label = "Line of Sufficient Decrease")
plt.scatter(alpha,f_x,label = "Pinpointing Points",facecolor = 'orange')
plt.xlabel('a')
plt.ylabel('f')
plt.legend()

#----------------------------------------Bean Function-----------------------------------------------------
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

    df_dx1 = -2 * (1 - x1) - 2 * x1 * (2 * x2 - x1**2)
    df_dx2 = -2 * (1 - x2) + (2 * x2 - x1**2)

    return np.vstack([df_dx1, df_dx2])

# Generate meshgrid
x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-2, 2, 400)
X1, X2 = np.meshgrid(x1, x2)
x = np.vstack([X1.ravel(), X2.ravel()])  # Stack to make 2xN matrix

# Evaluate the function
Z = bean_function(x)
Z = Z.reshape(X1.shape)  # Reshape back to the grid shape

# Contour plot
plt.figure()
plt.contour(X1, X2, Z, 50, cmap="jet")
plt.colorbar()
plt.title("Contour plot of the Bean Function")
plt.xlabel("x1")
plt.ylabel("x2")

# LineSearch_Backtracking
alpha_0 = 5
mu_1 = 1E-4
rho  = 0.7
x = np.array([[0],[-2]])
p_k = np.array([[1],[1]])

alpha, f_x, alpha_vec, phi_alpha_vec = LineSearch_BackTrack(alpha_0, mu_1, rho,bean_function,bean_function_gradient,x,p_k)

#Line Plot
plt.figure()
plt.plot(np.linspace(0,alpha_0,1000),bean_function(x + np.linspace(0,alpha_0,1000)*p_k),label = "Function Evaluation")
plt.scatter(alpha,f_x,label = "Back Tracking Points",facecolor = 'orange')
plt.xlabel('a')
plt.ylabel('Bean Function')
plt.title("x_0 = ["+str(x[0,0])+","+str(x[1,0])+"]. p_k = ["+str(p_k[0,0])+","+str(p_k[1,0])+"]")

sigma = 2
mu_2 = 0.05
a_init = 5
x_k = x

phi_0 = phi(bean_function,x_k, 0,p_k)
d_phi_0 = d_phi(bean_function_gradient, x_k, 0, p_k)

alpha, f_x = LineSearch_Bracketing (a_init, phi_0, d_phi_0, mu_1, mu_2, sigma, phi, d_phi,x_k,p_k,bean_function_gradient,bean_function)

plt.scatter(alpha,f_x,label = "Bracketing Points",facecolor = 'red')

plt.legend()


# LineSearch Bracketing



#---------------------------Rosenbrock Function-----------------------------
def rosenbrock(x):
    """
    x: A 2xN numpy array, where N is the number of column vectors.
    Returns: A 1xN numpy array representing the function value for each column vector.
    """
    x1 = x[0, :]
    x2 = x[1, :]
    return (1 - x1)**2 + 100 * (x2 - x1**2)**2

def rosenbrock_gradient(x):
    """
    x: A 2xN numpy array, where N is the number of column vectors.
    Returns: A 2xN numpy array representing the gradient for each column vector.
    """
    x1 = x[0, :]
    x2 = x[1, :]

    df_dx1 = -2 * (1 - x1) - 400 * x1 * (x2 - x1**2)
    df_dx2 = 200 * (x2 - x1**2)

    return np.vstack([df_dx1, df_dx2])

#---------Contour Plot-------------------------------------------------------
# Generate meshgrid
x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-3, 3, 400)
X1, X2 = np.meshgrid(x1, x2)
x = np.vstack([X1.ravel(), X2.ravel()])

# Evaluate the function
Z = rosenbrock(x)
Z = Z.reshape(X1.shape)

# Contour plot
plt.figure(figsize=(8, 6))
plt.contour(X1, X2, Z, 100, cmap="jet")
plt.colorbar()
plt.title("Contour plot of the Rosenbrock Function")
plt.xlabel("x1")
plt.ylabel("x2")


# LineSearch_Backtracking
alpha_0 = 10
mu_1 = 1E-4
rho  = 0.4
x = np.array([[-2],[2]])
p_k = np.array([[1],[1]])

alpha, f_x, alpha_vec, phi_alpha_vec = LineSearch_BackTrack(alpha_0, mu_1, rho,rosenbrock,rosenbrock_gradient,x,p_k)

#Line Plot
plt.figure()
plt.plot(np.linspace(0,alpha_0,1000),rosenbrock(x + np.linspace(0,alpha_0,1000)*p_k),label = "Function Evaluation")
plt.scatter(alpha,f_x,label = "Back Tracking Points",facecolor = 'orange')
plt.xlabel('a')
plt.ylabel('Rosenbrock Function 1D Slice')
plt.title("Backtracking Algorithm. x_0 = ["+str(x[0,0])+","+str(x[1,0])+"]. p_k = ["+str(p_k[0,0])+","+str(p_k[1,0])+"]")

sigma = 8
mu_2 = 0.2
a_init = 1
x_k = x

phi_0 = phi(rosenbrock,x_k, 0,p_k)
d_phi_0 = d_phi(rosenbrock_gradient, x_k, 0, p_k)

alpha, f_x = LineSearch_Bracketing (a_init, phi_0, d_phi_0, mu_1, mu_2, sigma, phi, d_phi,x_k,p_k,rosenbrock_gradient,rosenbrock)

plt.scatter(alpha,f_x,label = "Bracketing Points",facecolor = 'red')

plt.legend()

#----------------------------------------6D Rosenbrock---------------------------------------------------------
def rosenbrock_6D(x):
    """
    x: A 6xN numpy array, where N is the number of column vectors.
    Returns: A 1xN numpy array representing the function value for each column vector.
    """
    value = sum(100.0 * (x[i+1] - x[i]**2.0)**2.0 + (1 - x[i])**2.0 for i in range(5))
    return value

def rosenbrock_6D_gradient(x):
    """
    x: A 6xN numpy array, where N is the number of column vectors.
    Returns: A 6xN numpy array representing the gradient for each column vector.
    """
    grad = np.zeros_like(x)
    
    grad[0] = -400 * (x[1] - x[0]**2) * x[0] - 2 * (1 - x[0])
    
    for i in range(1, 5):
        grad[i] = 200 * (x[i] - x[i-1]**2) - 400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
    
    grad[5] = 200 * (x[5] - x[4]**2)
    
    return grad

# LineSearch_Backtracking
alpha_0 = 20
mu_1 = 1E-4
rho  = 0.7
x = np.array([[-10],[-9],[10],[3],[-1],[10]])
p_k = np.array([[1], [1], [1], [0], [1], [0]])

alpha, f_x, alpha_vec, phi_alpha_vec = LineSearch_BackTrack(alpha_0, mu_1, rho,rosenbrock_6D,rosenbrock_6D_gradient,x,p_k)

#Line Plot
plt.figure()
plt.plot(np.linspace(0,alpha_0,1000),rosenbrock_6D(x + np.linspace(0,alpha_0,1000)*p_k),label = "Function Evaluation")
plt.scatter(alpha,f_x,label = "Back Tracking Points",facecolor = 'orange')
plt.xlabel('a')
plt.ylabel('Rosenbrock Function 1D Slice')
plt.title("Backtracking Algorithm. x_0 = ["+str(x[0,0])+","+str(x[1,0])+","+str(x[2,0])+","+str(x[3,0])+","+str(x[4,0])+","+str(x[5,0])+"]. p_k = ["+str(p_k[0,0])+","+str(p_k[1,0])+","+str(p_k[2,0])+","+str(p_k[3,0])+","+str(p_k[4,0])+","+str(p_k[5,0])+"]")

sigma = 8
mu_2 = 0.2
a_init = 1
x_k = x

phi_0 = phi(rosenbrock_6D,x_k, 0,p_k)
d_phi_0 = d_phi(rosenbrock_6D_gradient, x_k, 0, p_k)

alpha, f_x = LineSearch_Bracketing (a_init, phi_0, d_phi_0, mu_1, mu_2, sigma, phi, d_phi,x_k,p_k,rosenbrock_6D_gradient,rosenbrock_6D)

plt.scatter(alpha,f_x,label = "Bracketing Points",facecolor = 'red')

plt.legend()

plt.show()
