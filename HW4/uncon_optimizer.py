"""
This is a template for Assignment 3: unconstrained optimization

You can (and should) call other functions or import functions from other files,
but make sure you do not change the function signature (i.e., function name `uncon_optimizer`, inputs, and outputs) in this file.
The autograder will import `uncon_optimizer` from this file. If you change the function signature, the autograder will fail.
"""

import numpy as np
import matplotlib.pyplot as plt
from LineSearch_Bracket import LineSearch_Bracketing_new
from LineSearch_BackTrack import LineSearch_BackTrack

def Uncon_BFGS(func, x0, epsilon_g, mu):
    """
    This algorithm implements the Quasi-Newton Unconstrained Optimization. It utilizes the
    backtracking line search strategy to ensure that the Strong Wolfe Condition is satisfied.
    """

    k = 0
    d = len(x0) #Detecting problem dimension

    g = 100 #Random value to enter loop
    x = x0
    while np.abs(np.max(g)) > epsilon_g: #While the maximum component of the gradient is still bigger than the tolerance, loop continues
        print(np.abs(np.max(g)))
        if k == 0: #First Iteration
            f,g = func(x, mu)  
            V_k = 1/(np.linalg.norm(g))*np.eye(d) #Assuming identity matrix times the magnitude of the gradient
        else:
            f_new,g_new = func(x, mu)  
            y = g_new - g
            g = g_new

            r = 1/(s.T@y)
            li = (np.eye(d)-(r*((s@(y.T)))))
            ri = (np.eye(d)-(r*((y@(s.T)))))
            hess_inter = li@V_k@ri
            V_k = hess_inter + (r*((s@(s.T)))) # BFGS Update

        alpha_0 = 0.01
        mu_1 = 1E-6
        rho  = 0.8
        
        p = -V_k@g
        mu_1 = 1E-4
        mu_2 = 0.9
        LineSearchResult = LineSearch_Bracketing_new(alpha_0, func, mu_1,mu_2,1.1,x,p,mu)
        alpha = LineSearchResult[0]
        s = alpha*p
        x = x + s
        k = k + 1

    f_finale,g_finale = func(x, mu) 
    return x, f_finale, g_finale



def uncon_optimizer(func, x0, epsilon_g, mu, options=None):
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

        f, g = func(x0, mu)
        dim = g.size
        g_vec = np.array([np.max(np.abs(g))])
        x = x0
        x_k_vec = np.array(x)    

        while np.abs(np.max(g)) > epsilon_g:
            if k == 0: #First Iteration
                V_k = 1/(np.linalg.norm(g))*np.eye(dim)
            else:
                s = alpha*p
                
  

            p = -V_k@g

            mu_1 = 1E-4
            mu_2 = 0.9
            LineSearchResult = LineSearch_Bracketing_new(alpha_init, func, mu_1,mu_2,2,x_curr,p)
            alpha = LineSearchResult[0]
            #print(alpha)
            x_new = x + np.dot(alpha,p)
            x_k_vec = np.hstack((x_k_vec,x_next))

            V_k_prev = V_k

            x_prev = x_curr
            x_curr = x_next

            f,g = func(x_curr, mu)
            g_vec = np.append(g_vec,np.max(np.abs(g)))

            k = k+1

        print(V_k)
        xopt = x_curr 
        fopt = func(xopt,mu)
        fopt = fopt[0]

    elif options == "SD OUTPUT-ALL" or options is None:
        #Steepest Descent Implementation
        x_k = x0
        f,g = func(x0,mu)
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
            f,g = func(x_k,mu)
            #print(np.max(np.abs(g)))
            mag_f_k = np.linalg.norm(g)
            
            p_k = -g/mag_f_k

            a_k = 1E-6

            LineSearchResult = LineSearch_Bracketing_new(a_k, func, 1E-4 , 0.1, 2,x_k,p_k, mu)

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
            #print(np.max(np.abs(g)))



        xopt = x_k 
        fopt = func(xopt, mu)
        fopt = fopt[0]

    if output_All:
        return xopt, fopt, output, x_k_vec, g_vec

    return xopt, fopt, output

#Result = uncon_optimizer(bean_function_ALL,np.array([[-1],[2]]) , 1E-6, options=None)

#print (Result[0])
#print (Result[1])


