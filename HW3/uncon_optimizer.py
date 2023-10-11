"""
This is a template for Assignment 3: unconstrained optimization

You can (and should) call other functions or import functions from other files,
but make sure you do not change the function signature (i.e., function name `uncon_optimizer`, inputs, and outputs) in this file.
The autograder will import `uncon_optimizer` from this file. If you change the function signature, the autograder will fail.
"""

import numpy as np
import matplotlib.pyplot as plt
from LineSearch_Bracket import LineSearch_Bracketing_new

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


def uncon_optimizer(func, x0, epsilon_g, options=None):
    """An algorithm for unconstrained optimization.

    Parameters
    ----------
    func : function handle
        Function handle to a function of the form: f, g = func(x)
        where f is the function value and g is a numpy array containing
        the gradient. x are design variables only.
    x0 : ndarray
        Starting point
    epsilon_g : float
        Convergence tolerance.  you should terminate when
        np.max(np.abs(g)) <= epsilon_g.  (the infinity norm of the gradient)
    options : dict
        A dictionary containing options.  You can use this to try out different
        algorithm choices.  I will not pass anything in on autograder,
        so if the input is None you should setup some defaults.

    Returns
    -------
    xopt : ndarray
        The optimal solution
    fopt : float
        The corresponding optimal function value
    output : dictionary
        Other miscelaneous outputs that you might want, for example an array
        containing a convergence metric at each iteration.

        `output` must includes the alias, which will be used for mini-competition for extra credit.
        Do not use your real name or uniqname as an alias.
        This alias will be used to show the top-performing optimizers *anonymously*.
    """

    # TODO: set your alias for mini-competition here
    output = {}
    output['alias'] = 'A'

    output_All = False

    if options is None:
        # TODO: set default options here.
        # You can pass any options from your subproblem runscripts, but the autograder will not pass any options.
        # Therefore, you should sse the  defaults here for how you want me to run it on the autograder.
        True

    # TODO: Your code goes here!
    elif options == "BFGS OUTPUT-ALL" or options == "SD OUTPUT-ALL":
        output_All = True
    
    if options == "BFGS OUTPUT-ALL":
        k = 0
        alpha_init = 1 #Initial Steps size set to 1

        f, g = func(x0)
        dim = g.size
        g_vec = np.array([np.max(np.abs(g))])
        x_curr = x0
        x_prev = 0 #
        x_k_vec = np.array(x_curr)    

        while np.abs(np.max(g)) > epsilon_g:
            if k == 0:
                V_k = 1/(np.linalg.norm(g))*np.eye(dim)
            else:
                s = x_curr - x_prev

                sol_curr = func(x_curr)
                sol_prev = func(x_prev)

                y = sol_curr[1] - sol_prev[1]

                sigma = 1/(np.dot(s.T,y))

                V_k_1 = (np.eye(dim) - np.dot(sigma*s,y.T))
                V_k_2 = (np.eye(dim) - np.dot(sigma*y,s.T))
                V_k_3 = np.dot(sigma*s,s.T)

                V_k = np.dot(np.dot(V_k_1,V_k_prev),V_k_2) + V_k_3

            p = -np.dot(V_k,g)

            mu_1 = 1E-4
            mu_2 = 0.9
            LineSearchResult = LineSearch_Bracketing_new(alpha_init, func, mu_1,mu_2,2,x_curr,p)
            alpha = LineSearchResult[0]
            #print(alpha)
            x_next = x_curr + np.dot(alpha,p)
            x_k_vec = np.hstack((x_k_vec,x_next))

            V_k_prev = V_k

            x_prev = x_curr
            x_curr = x_next

            f,g = func(x_curr)
            g_vec = np.append(g_vec,np.max(np.abs(g)))

            k = k+1

        print(V_k)
        xopt = x_curr 
        fopt = func(xopt)
        fopt = fopt[0]

    elif options == "SD OUTPUT-ALL" or options is None:
        #Steepest Descent Implementation
        x_k = x0
        f,g = func(x0)
        k = 0
        g_vec = np.array([np.max(np.abs(g))])

        """
        x1 = np.linspace(-4, 4, 400)
        x2 = np.linspace(-4, 4, 400)
        X1, X2 = np.meshgrid(x1, x2)
        x = np.vstack([X1.ravel(), X2.ravel()])  # Stack to make 2xN matrix

        # Evaluate the function
        Z = bean_function(x)
        Z = Z.reshape(X1.shape)  # Reshape back to the grid shape


        fig, axs = plt.subplots(2)
        axs[1].contour(X1, X2, Z, 80, cmap="jet")
        axs[1].axis("equal")
        """

        x_k_vec = np.array(x_k)    
        while np.max(np.abs(g)) > epsilon_g:
            f,g = func(x_k)
            #print(np.max(np.abs(g)))
            mag_f_k = np.linalg.norm(g);
            
            p_k = -g/mag_f_k

            a_k = 1

            LineSearchResult = LineSearch_Bracketing_new(a_k, func, 1E-4 , 0.1, 2,x_k,p_k)

            #Plotting

            """
            plot_eval = func(x_k + np.linspace(0,np.max(LineSearchResult[2]),1000)*p_k)
            axs[0].plot(np.linspace(0,np.max(LineSearchResult[2]),1000),plot_eval[0],label = "Function Evaluation")
            axs[0].scatter(LineSearchResult[2],LineSearchResult[3],label = "Bracketing Points")
            axs[0].scatter(LineSearchResult[4],LineSearchResult[5],label = "Pinpointing Points")
            axs[0].legend()

            nextPoint = x_k + LineSearchResult[0]*p_k
            axs[1].plot([x_k[0], nextPoint[0]], [x_k[1], nextPoint[1]], 'b-')  # 'b-' means blue color, solid line
            plt.scatter([x_k[0], nextPoint[0]], [x_k[1], nextPoint[1]], color='red')

            plt.show()

            axs[0].clear()
            # Optionally, plot the points themselves
            """

            x_k = x_k + LineSearchResult[0]*p_k
            x_k_vec = np.hstack((x_k_vec,x_k))

            k = k+1
            # print(x_k)

            g_vec = np.append(g_vec,np.max(np.abs(g)))



        xopt = x_k 
        fopt = func(xopt)
        fopt = fopt[0]

    if output_All:
        return xopt, fopt, output, x_k_vec, g_vec

    return xopt, fopt, output

#Result = uncon_optimizer(bean_function_ALL,np.array([[-1],[2]]) , 1E-6, options=None)

#print (Result[0])
#print (Result[1])


