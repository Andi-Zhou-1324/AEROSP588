import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from LineSearch_Bracket_SQP import LineSearch_Bracketing_new

# This function reads in and calculates the Jacobian symbolically regardless of dimensions
def calculate_jacobian(constraints, num_dims):
    # Create symbolic variables
    variables = [sp.symbols(f'x{i+1}') for i in range(num_dims)]
    
    # Initialize Jacobian matrix
    jacobian = []
    
    # Populate the Jacobian matrix
    for constraint in constraints:
        partial_derivatives = [sp.diff(constraint, var) for var in variables]
        jacobian.append(partial_derivatives)
    
    # Convert symbolic matrix to numpy array
    lambdified_jacobian = sp.lambdify(variables, jacobian, 'numpy')
    
    return lambdified_jacobian

def rosenbrock_sym(N):
    """
    Compute the N-dimensional Rosenbrock function.

    Parameters:
        - x: List or array of values, representing a point in N-dimensional space.

    Returns:
        - Value of the Rosenbrock function at the given point.
    """
    return sum(100.0*(x[i+1] - x[i]**2.0)**2.0 + (1 - x[i])**2.0 for i in range(len(x)-1))

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





def func (x):
    return f_num(*x), np.array(d_f_num(*x)).reshape(-1,1)

def jac_func(x,n_h):
    ans = jacobian_func(*x)
    arr = np.array([item[0] if isinstance(item, np.ndarray) else item for inner_lst in ans for item in inner_lst])

    return np.array(arr).reshape(n_h,-n_h)

x_0 = np.array([[2.5],[5]])

#print(func(x_0))


#CHECK THIS FUNCTION WHEN WE HAVE MULTIPLE CONSTRAINTS
def evaluate_functions(funcs, args):
    """
    Evaluate a list of lambdified functions with given arguments.

    Parameters:
    - funcs: List of lambdified functions
    - *args: Variable length argument list of inputs to the functions

    Returns:
    - numpy array of function outputs
    """
    results = [f(*args) for f in funcs]
    return np.array([results]).reshape(1, -1).T



def analytical_merit_gradient(f, h, dim):
    # Create symbolic variables based on the given dimensions
    n_h = len(h)
    lambda_i = [sp.symbols(f'lambda_{i+1}') for i in range(n_h)]
    vars = [sp.symbols(f'x{i+1}') for i in range(dim)]
    mu   = sp.symbols('mu')

    term_1 = sum([h[i]*lambda_i[i] for i in range(n_h)])
    term_2 = mu/2*sum([h[i]**2 for i in range(n_h)])

    merit_func = f + term_1 + term_2

    # Compute the gradient of the L2 norm
    gradient = [sp.diff(merit_func, var) for var in vars]

    #Convert to merit gradient function
    gradient_func = sp.lambdify((*vars, *lambda_i, mu), gradient, 'numpy')

    merit_func_Lambda = sp.lambdify((*vars,*lambda_i, mu), merit_func,'numpy')

    return merit_func_Lambda, gradient_func


def merit (x,mu, Lambda):
    merit          = merit_func(*x,*Lambda, mu)
    merit_gradient = merit_func_gradient(*x, *Lambda,mu)
    return merit, np.array([item[0] for item in merit_gradient]).reshape(-1, 1)

#print(merit(x0,1.5))



def Quasi_Newton_SQP (x0, func, jac_func, tau_opt, tau_feas, h_func):
    x = x0
    n_x = len(x)

    n_h = len(h_func)
    Lambda = np.zeros((n_h,1)) #Number of Lambda is scaled by the number of problem constraints
    alpha_init = 1
    
    h = evaluate_functions(h_func,x)
    J_h = jac_func(x, n_h)
    f,d_f = func(x)
    A_dim = n_x + n_h
    grad_L = d_f + J_h.T@Lambda
    k = 0

    x_vec = np.array(x)   #Records history of x_vec
    df_vec= np.array(d_f) #Records history of gradient

    while np.abs(np.max(grad_L)) > tau_opt or np.abs(np.max(h)) > tau_feas:
        if k == 0:
            H_L = np.eye(n_x)
        else:
            if s_k.T@y_k >= 0.2*s_k.T@H_L@s_k:
                theta_k = 1
            else:
                theta_k = 0.8*s_k.T@H_L@s_k/(s_k.T@H_L@s_k - s_k.T@y_k)

            r_k = theta_k*y_k + (1-theta_k)*H_L@s_k

            term_1 = H_L@s_k@s_k.T@H_L/(s_k.T@H_L@s_k)
            term_2 = r_k@r_k.T/(r_k.T@s_k)

            H_L = H_L - term_1 + term_2

        #We now proceed to solve the equality-constrained SQP problem
        A = np.zeros((A_dim,A_dim))
        A[0:n_x,0:n_x] = H_L
        A[n_x:n_x + n_h,0:n_x] = J_h
        A[0:n_x,n_x:n_x+n_h] = J_h.T

        B = -np.vstack((grad_L,h))

        p = np.linalg.solve(A,B)
        p_x = p[0:n_x]

        p_Lambda = p[n_x:n_x+n_h]
        Lambda = Lambda + p_Lambda

        #Step for Linesearch
        mu_1 = 1E-4
        mu_2 = 0.9
        mu = 1 #Merit factor
        LineSearchResult = LineSearch_Bracketing_new(alpha_init, merit, mu_1,mu_2,1.2,x,p_x, mu, Lambda)
        alpha = LineSearchResult[0]
        s_k = alpha*p_x
        x = x + s_k

        
        h = evaluate_functions(h_func,x) #Evaluating h to check for feasbility constraint
        J_h = jac_func(x, n_h)
        f,d_f = func(x)


        x_vec = np.hstack((x_vec, x))
        df_vec = np.hstack((df_vec, d_f))


        k = k + 1

        #Evaluate Lagrangian gradient at the last point with current Lagrangian multiplier
        J_h_prev = jac_func(x_vec[:,k-1:k], n_h)
        f_prev,d_f_prev = func(x_vec[:,k-1:k])

        grad_L = d_f + J_h.T@Lambda
        grad_L_prev = d_f_prev + J_h_prev.T@Lambda

        y_k = grad_L - grad_L_prev

        #print(np.abs(np.max(grad_L)))
        #print(np.abs(np.max(h)))
        #print(x, k)

    return x, x_vec, k



def rosenbrock_meshgrid(X, Y):
    """Evaluate the Rosenbrock function for meshgrid inputs."""
    return 100.0 * (Y - X**2.0)**2.0 + (1 - X)**2.0


def plot_constraints(h):
    x = [sp.symbols(f'x{i+1}') for i in range(2)]

    x_lim = np.linspace(-2, 2, 400)
    y_lim = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x_lim, y_lim)
    
    
    for func in h:
        # Create a lambda function for vectorized evaluation
        f_lambdified = sp.lambdify((x[0], x[1]), func, "numpy")
        Z = f_lambdified(X, Y)
        
        # Plot the contour where the function value is zero (i.e., the constraint boundary)
        plt.contour(X, Y, Z, levels=[0], colors='b')

D_vec = np.array([2])
k_vec = np.array([])
for D in D_vec:
    x = [sp.symbols(f'x{i+1}') for i in range(D)]
    mu, Lambda = sp.symbols('mu, Lambda')

    h = [x[0]**2 + x[1]**2 - 2] #Varies with Dimension
 #   h = [x[0]**2 + x[1]**2 - 2, 2*x[1] - 3*x[3] + x[4]] #Varies with Dimension
 #   h = [x[0]**2 + x[1]**2 - 2, 2*x[1] - 3*x[3] + x[4], x[0] + x[2] - x[3] + x[5] - x[6] - 2] #Varies with Dimension
 #   h = [x[0]**2 + x[1]**2 - 2, 2*x[1] - 3*x[3] + x[4], x[0] + x[2] - x[3] + x[5] - x[6] - 2, x[4] + x[5] + x[6] - x[7] - x[8]] #Varies with Dimension
 #   h = [x[0]**2 + x[1]**2 - 2, 2*x[1] - 3*x[3] + x[4], x[0] + x[2] - x[3] + x[5] - x[6] - 2, x[4] + x[5] + x[6] - x[7] - x[8],x[1] + x[3] - x[5] + x[7] - x[9] - 3 ] #Varies with Dimension

    jacobian_func = calculate_jacobian(h, D)

    f   = rosenbrock_sym(D)
    d_f = [sp.diff(f, i) for i in x]

    f_num = sp.lambdify((x),f,'numpy')
    d_f_num = sp.lambdify((x),d_f,'numpy')

    h_func = [sp.lambdify((x),h,"numpy") for h in h]

    merit_func, merit_func_gradient = analytical_merit_gradient(f, h, D)

    x0 = np.ones((D,1))*(-5)


    tau_opt = 1E-6
    tau_feas = 1E-6

    result = Quasi_Newton_SQP (x0, func, jac_func, tau_opt, tau_feas, h_func)
    k = result[2]
    k_vec = np.hstack((k_vec,k))

    #print(result[0], k)


def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

# Constraint function (applies for any dimension, but only restricts x1 and x2)
def eq_constraint1(x):
    return 2 - x[0]**2 - x[1]**2

# Equality Constraint 2
def eq_constraint2(x):
    return 2*x[1] - 3*x[3] + x[4]

# Equality Constraint 3
def eq_constraint3(x):
    return x[0] + x[2] - x[3] + x[5] - x[6] - 2

# Equality Constraint 4
def eq_constraint4(x):
    return x[4] + x[5] + x[6] - x[7] - x[8]

# Equality Constraint 5
def eq_constraint5(x):
    return x[1] + x[3] - x[5] + x[7] - x[9] - 3

# Define the constraint (g(x) >= 0 form)
cons = [{'type': 'eq', 'fun': eq_constraint1}]
#cons = [{'type': 'eq', 'fun': eq_constraint1},{'type': 'eq', 'fun': eq_constraint2}]
#cons = [{'type': 'eq', 'fun': eq_constraint1},{'type': 'eq', 'fun': eq_constraint2},{'type': 'eq', 'fun': eq_constraint3}]
#cons = [{'type': 'eq', 'fun': eq_constraint1},{'type': 'eq', 'fun': eq_constraint2},{'type': 'eq', 'fun': eq_constraint3},{'type': 'eq', 'fun': eq_constraint4}]
#cons = [{'type': 'eq', 'fun': eq_constraint1},{'type': 'eq', 'fun': eq_constraint2},{'type': 'eq', 'fun': eq_constraint3},{'type': 'eq', 'fun': eq_constraint4},{'type': 'eq', 'fun': eq_constraint5}]



def optimize_rosenbrock(dim):
    # Initial guess
    x0 = [-5] * dim

    # Perform minimization using SLSQP (Quasi-Newton SQP)
    res = minimize(rosenbrock, x0, constraints=cons, method='SLSQP')
    print(res.x)
    return res
# Example usage

k_vec_scipy = np.array([])

for D in D_vec:
    res = optimize_rosenbrock(D)
    it = res.nit
    k_vec_scipy = np.hstack((k_vec_scipy,it))

print(it)
num_contraints = np.array([1,2,3,4,5])
k_vec_h = np.array([158,148,128,61,52])
k_vec_h_SciPy = np.array([100,100,100,59,48])

"""
plt.figure()
plt.plot(num_contraints ,k_vec_h, label = 'Quasi-Newton SQP')
plt.plot(num_contraints ,k_vec_h_SciPy, label = 'SciPy SLSQP')
plt.xlabel('Number of Constraints')
plt.ylabel('Iteration')
plt.grid()
plt.legend()
plt.show()
"""
"""
plt.figure()
plt.plot(D_vec, k_vec, label = 'Quasi-Newton SQP')
plt.plot(D_vec, k_vec_scipy, label = 'SciPy SLSQP')
plt.xlabel('Dimension')
plt.ylabel('Iteration')
plt.grid()
plt.legend()
plt.show()

"""

"""
x = np.linspace(-np.max(np.max(x_vec)), np.max(np.max(x_vec)), 400)
y = np.linspace(-np.max(np.max(x_vec)), np.max(np.max(x_vec)), 400)
X, Y = np.meshgrid(x, y)

# Compute the value of the function on the grid
F = rosenbrock_meshgrid(X, Y)

# Plot the contour line where the function value is 0
plot_constraints(h)
plt.contourf(X, Y, F)  # plotting contour for the value 0
plt.colorbar()
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.plot(x_vec[0,:],x_vec[1,:], color = 'cyan',label = 'Optimization Path')
plt.scatter(x_vec[0,:],x_vec[1,:], color = 'cyan')
plt.legend()
plt.axis('equal')
plt.grid(True)




plt.show()
"""