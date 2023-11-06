import numpy as np

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

def func(x, mu):
    return f(x), d_f(x)



def LineSearch_BackTrack(alpha_0, mu_1, rho,func,x,p_k,mu):
    #This function takes in an initial step size used for backtracking.
    #mu_1 is the constant that specify the size of the decrease that is deemed
    #as enough. rho is the backtracking factor. f is the original function, while
    #d_f is its gradient information, specified as a column vector.
    #x is the current position, in column vector
    #p_k is the search direction, in column vector
    alpha = alpha_0

    f,g = func(x,mu)

    phi_0 = f
    d_phi_0 = g.T@p_k

    phi_alpha, d_phi_alpha = func(x + alpha*p_k, mu)

    alpha_vec = np.array([alpha])
    phi_alpha_vec = np.array([phi_alpha])
    it = 0
    while phi_alpha >= phi_0 + mu_1*alpha*d_phi_0:
        alpha = rho*alpha
        phi_alpha, d_phi_alpha = func(x + alpha*p_k, mu)

        alpha_vec = np.append(alpha_vec,alpha)
        phi_alpha_vec = np.append(phi_alpha_vec,phi_alpha)
        it += 1

        if it > 10:
            break
            print ("Iteration Exceeded 10. Force Exit")

    print("Backtrack Line Search Successful")
    return alpha, phi_alpha, alpha_vec, phi_alpha_vec

"""
alpha_0 = 1.2
mu_1    = 1E-4
rho     = 0.7
x       = np.array([[-1.25],[1.25]])
p_k     = np.array([[4],[0.75]])
mu      = 0

result = LineSearch_BackTrack(alpha_0, mu_1, rho,func,x,p_k,mu)
print(result)
"""