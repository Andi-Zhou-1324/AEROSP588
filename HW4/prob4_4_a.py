import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from LineSearch_Bracket_SQP import LineSearch_Bracketing_new
"""
In this comment, we take some time and discuss how to implement a Quasi-Newton SQP Optimizer
1. 

We first define the optimality conditions and the feasibility conditions. The optimality condition is
defined as the maximum value within the Lagrangian gradient. The feasbility condition is defined as the
maximum value within the constraint

We start by assuming the Hessian as an identity matrix or a equivalent scaled version.
We then seek to solve the linear quadratic optimization problem, which we could do with np.linalg package
We would obtain the step direction and the Lagrangian update. We have to input the search direction into our
augmented line search algorithm, which we use a MERIT function as the function for line search
We then update our step
And evaluate the functions and gradients again. Repeat

On the second iteration, we update the Hessian using the damped Hessian update given

Major Components to Code:
Jacobian extractor
Linear Solver for step
Merit function

For this problem we always assume EQUALITY CONSTRAINT. Therefore, no active sets
"""
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



D = 2
x = [sp.symbols(f'x{i+1}') for i in range(D)]
mu, Lambda = sp.symbols('mu, Lambda')

h = [((1/4) * x[0]**2 + x[1]**2 - 1)] #Varies with Dimension
jacobian_func = calculate_jacobian(h, D)

f   = x[0] + 2*x[1]
d_f = [sp.diff(f, i) for i in x]

f_num = sp.lambdify((x),f,'numpy')
d_f_num = sp.lambdify((x),d_f,'numpy')

h_func = [sp.lambdify((x),h,"numpy") for h in h]

def jac_func(x):
    x1 = x[0,0]
    x2 = x[1,0]
    ans = np.array(jacobian_func(x1,x2))
    return ans

def func (x):
    x1 = x[0,0]
    x2 = x[1,0]
    return f_num(x1,x2), np.array([d_f_num(x1,x2)]).T


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
    return np.array(results).T



def analytical_merit_gradient(f, h, dim):
    # Create symbolic variables based on the given dimensions
    n_h = len(h)
    lambda_i = [sp.symbols(f'lambda_{i+1}') for i in range(n_h)]
    vars = [sp.symbols(f'x{i+1}') for i in range(dim)]
    mu   = sp.symbols('mu')

    term_1 = sum([h[i]*lambda_i[i] for i in range(n_h)])
    term_2 = mu/2*sum([h[i]**2 for i in range(n_h)])

    # For demonstration purposes, I'm assuming the expressions are simply the variables themselves.
    # You can replace this with any list of symbolic expressions based on these variables.


    # Compute the L1 norm of the list of expressions

    l2_norm = sp.sqrt(sum((expr**2) for expr in h))


    merit_func = f + term_1 + term_2

    # Compute the gradient of the L2 norm
    gradient = [sp.diff(merit_func, var) for var in vars]

    #Convert to merit gradient function
    gradient_func = sp.lambdify((*vars, *lambda_i, mu), gradient, 'numpy')

    merit_func_Lambda = sp.lambdify((*vars,*lambda_i, mu), merit_func,'numpy')

    return merit_func_Lambda, gradient_func

merit_func, merit_func_gradient = analytical_merit_gradient(f, h, D)

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
    J_h = jac_func(x)
    f,d_f = func(x)
    A_dim = n_x + n_h
    grad_L = d_f + J_h.T@Lambda
    k = 0

    x_vec = np.array(x)   #Records history of x_vec
    h_vec = np.array(np.abs(np.max(h)))
    grad_L_vec = np.array(np.abs(np.max(grad_L)))
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
        mu = 0.01 #Merit factor
        LineSearchResult = LineSearch_Bracketing_new(alpha_init, merit, mu_1,mu_2,1.2,x,p_x, mu, Lambda)
        alpha = LineSearchResult[0]
        s_k = alpha*p_x
        x = x + s_k

        
        h = evaluate_functions(h_func,x) #Evaluating h to check for feasbility constraint
        J_h = jac_func(x)
        f,d_f = func(x)


        x_vec = np.hstack((x_vec, x))
        df_vec = np.hstack((df_vec, d_f))
        h_vec = np.hstack((h_vec, np.abs(np.max(h))))
        grad_L_vec = np.hstack((grad_L_vec, np.abs(np.max(grad_L))))

        k = k + 1

        #Evaluate Lagrangian gradient at the last point with current Lagrangian multiplier
        J_h_prev = jac_func(x_vec[:,k-1:k])
        f_prev,d_f_prev = func(x_vec[:,k-1:k])

        grad_L = d_f + J_h.T@Lambda
        grad_L_prev = d_f_prev + J_h_prev.T@Lambda

        y_k = grad_L - grad_L_prev



        print(np.abs(np.max(grad_L)))
        #print(np.abs(np.max(h)))
        #print(x, k)

    return x_vec, h_vec, grad_L_vec




x0 = np.array([[2],[1]])



tau_opt = 1E-6
tau_feas = 1E-6

result = Quasi_Newton_SQP (x0, func, jac_func, tau_opt, tau_feas, h_func)
#print(x_vec)
x_vec = result[0]
plt.figure()
plt.plot(result[1])
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Feasibility")
plt.grid()

plt.figure()
plt.plot(result[2])
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Optimality")
plt.grid()

plt.figure()
x = np.linspace(-np.max(np.max(np.abs(x_vec))), np.max(np.max(np.abs(x_vec))), 400)
y = np.linspace(-np.max(np.max(np.abs(x_vec))), np.max(np.max(np.abs(x_vec))), 400)
X, Y = np.meshgrid(x, y)

# Compute the value of the function on the grid
F = X + 2*Y

h = (1/4)*X**2 + Y**2 - 1

# Plot the contour line where the function value is 0


plt.contourf(X, Y, F)  # plotting contour for the value 0
plt.colorbar()
plt.contour(X, Y, h, levels=[0], colors='blue',zorder = 2)

plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.plot(x_vec[0,:],x_vec[1,:], color = 'cyan',label = 'Optimization Path')
plt.scatter(x_vec[0,:],x_vec[1,:], color = 'cyan')
plt.scatter(x_vec[0,-1],x_vec[1,-1], color = 'red')

plt.legend()
plt.grid(True)




plt.show()
