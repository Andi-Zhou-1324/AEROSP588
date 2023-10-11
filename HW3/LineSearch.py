import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

#Importing needed libraries

def LineSearch_BackTrack(alpha_0, mu_1, rho,f,d_f,x,p_k):
    #This function takes in an initial step size used for backtracking.
    #mu_1 is the constant that specify the size of the decrease that is deemed
    #as enough. rho is the backtracking factor. f is the original function, while
    #d_f is its gradient information, specified as a column vector.
    #x is the current position, in column vector
    #p_k is the search direction, in column vector
    alpha = alpha_0

    phi_0 = f(x)
    d_phi_0 = np.dot(d_f(x).T,p_k)

    phi_alpha = f(x + alpha*p_k)

    alpha_vec = np.array([alpha])
    phi_alpha_vec = np.array([phi_alpha])

    while phi_alpha >= phi_0 + mu_1*alpha*d_phi_0:
        alpha = rho*alpha
        phi_alpha = f(x + alpha*p_k)

        alpha_vec = np.append(alpha_vec,alpha)
        phi_alpha_vec = np.append(phi_alpha_vec,phi_alpha)

    print("Backtrack Line Search Successful")
    return alpha, phi_alpha, alpha_vec, phi_alpha_vec

def phi (func,x_k, a,p_k):
    return (func(x_k + a*p_k))

def d_phi(d_func, x_k, a, p_k):
    return np.dot(d_func(x_k + a*p_k).T,p_k)

def Quad_Interp(a_high, a_low, phi_high, phi_low, d_phi_low):
    numerator = 2 * a_low * (phi_high - phi_low) + d_phi_low * (a_low**2 - a_high**2)
    denominator = 2 * (phi_high - phi_low + d_phi_low * (a_low - a_high))
    
    return numerator / denominator   

def Pinpoint (a_1,a_2,phi_0,phi_1,phi_2,d_phi_0,d_phi_1,d_phi_2,mu_1,mu_2,f,d_f,x_k,p_k):
    #Identifies the lowest and high value, and classify them accordingly
    if phi_1 < phi_2:
        a_low = a_1
        a_high = a_2
        phi_low = phi_1
        phi_high = phi_2
        d_phi_low = d_phi_1
        d_phi_high = d_phi_2
    else:
        a_low = a_2
        a_high = a_1
        phi_low = phi_2
        phi_high = phi_1
        d_phi_low = d_phi_2
        d_phi_high = d_phi_1

    k = 0
    a_p_vec = np.array([])
    phi_p_vec = np.array([])
    while True:
        a_p = Quad_Interp(a_high, a_low, phi_high, phi_low, d_phi_low) #Obtain new point via interpolation
        a_p = a_p.item()
        a_p_vec = np.append(a_p_vec,a_p)

        phi_p = phi(f,x_k,a_p,p_k) #calculating function values at a_p
        phi_p_vec = np.append(phi_p_vec,phi_p)
        d_phi_p = d_phi(d_f,x_k,a_p,p_k) #calculating function gradient at a_p

        #if the new point is above the sufficient decrease line or higher than the low point
        if phi_p > phi_0 + mu_1*a_p*d_phi_0 or phi_p > phi_low:
            a_high = a_p
            phi_high = phi_p
        else:
            #If the slope at new point is already small than tolerance
            if np.abs(d_phi_p) <= -mu_2*d_phi_0:
                a_star = a_p
                return a_star, phi_p, a_p_vec, phi_p_vec
            #If the slope at new point is positive
            elif d_phi_p*(a_high - a_low) >= 0:
                a_high = a_low

            a_low = a_p
        k = k + 1


def LineSearch_Bracketing (a_init,phi_0,d_phi_0,mu_1,mu_2,sigma,phi,d_phi,x_k,p_k,d_f,f):
    a_1 = 0
    a_2 = a_init
    phi_1 = phi_0
    d_phi_1 = d_phi_0
    first = True

    while True:
        #Function evaluation at point a_2
        phi_2 = phi(f,x_k,a_2,p_k)

        #Derivative at point a_2
        d_phi_2 = d_phi(d_f, x_k, a_2, p_k)
        
        #If slope is positive and satisfies suifficient decrease, or the second step is higher than the first
        #We have bracketed a minimum. Note that the initial direction p_k is always in descent direction

        if (phi_2 > phi_0 + mu_1*a_2*d_phi_0) or (not first and phi_2 > phi_1):
            a_star, phi_star, a_p_vec, phi_p_vec = Pinpoint(a_1,a_2,phi_0,phi_1,phi_2,d_phi_0,d_phi_1,d_phi_2,mu_1,mu_2,f,d_f,x_k,p_k)
            return a_star, phi_star

        #If the derivative at step 2 is flat enough, then this step is acceptable
        if np.abs(d_phi_2) < -mu_2*d_phi_0:
            a_star = a_2
            phi_star = phi(f,x_k,a_star,p_k)
            return a_star, phi_star
        elif d_phi_2 >= 0: #Else, if the derivative at the second step is positive, then we have also bracketed a minimum
            a_star, phi_star, a_p_vec, phi_p_vec = Pinpoint(a_1,a_2,phi_0,phi_1,phi_2,d_phi_0,d_phi_1,d_phi_2,mu_1,mu_2,f,d_f,x_k,p_k)
            return a_star, phi_star
        else: #No minimum bracketed, keep expanding search
            a_1 = a_2
            a_2 = sigma*a_2

        first = False

