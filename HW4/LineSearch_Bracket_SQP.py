import numpy as np


def phi(func,x_k, a,p_k, mu, Lambda):
    f,g = (func(x_k + a*p_k, mu, Lambda))
    return f,np.dot(g.T,p_k)



def LineSearch_Bracketing_new(a_init, func,mu_1,mu_2,sigma,x_k,p_k, mu, Lambda):
    a_1 = 0
    a_2 = a_init
    
    phi_0, d_phi_0 = phi(func, x_k, 0, p_k, mu, Lambda)

    phi_1 = phi_0
    d_phi_1 = d_phi_0
    first = True

    a_array = np.array([])
    phi_array = np.array([])

    a_pinpoint_array = np.array([])
    phi_pinpoint_array = np.array([])

    while True:
        phi_2, d_phi_2 = phi(func, x_k, a_2, p_k, mu, Lambda)

        if not first:
            phi_1, d_phi_1 = phi(func, x_k, a_1, p_k, mu, Lambda)

        a_array = np.append(a_array,a_2)
        phi_array = np.append(phi_array,phi_2)

        if (phi_2 > phi_0 + mu_1*a_2*d_phi_0) or (not first and phi_2 > phi_1):
            a_star,phi_star, a_pinpoint_array, phi_pinpoint_array = Pinpoint (a_1,a_2,phi_0,phi_1,phi_2,d_phi_0,d_phi_1,d_phi_2,mu_1,mu_2,func,x_k,p_k, mu, Lambda)
            return a_star, phi_star, a_array, phi_array, a_pinpoint_array, phi_pinpoint_array
        
        if np.abs(d_phi_2) < -mu_2*d_phi_0:
            a_star = a_2
            phi_star = phi(func,x_k,a_star,p_k, mu, Lambda)
            phi_star = phi_star[0]
            return a_2, phi_star, a_array, phi_array, a_pinpoint_array, phi_pinpoint_array
        
        elif  d_phi_2 >= 0:
            a_star, phi_star, a_pinpoint_array, phi_pinpoint_array = Pinpoint (a_2,a_1,phi_0,phi_2,phi_1,d_phi_0,d_phi_2,d_phi_1,mu_1,mu_2,func,x_k,p_k, mu, Lambda)
            return a_star, phi_star, a_array, phi_array, a_pinpoint_array, phi_pinpoint_array
        else:
            a_1 = a_2
            a_2 = sigma*a_2

        first = False


def Pinpoint (a_low,a_high,phi_0,phi_low,phi_high,d_phi_0,d_phi_low,d_phi_high,mu_1,mu_2,func,x_k,p_k, mu, Lambda):
    #Identifies the lowest and high value, and classify them accordingly

    k = 0
    a_p_vec = np.array([])
    phi_p_vec = np.array([])
    while True:
        a_p = Quad_Interp(a_high, a_low, phi_high, phi_low, d_phi_low) #Obtain new point via interpolation
        a_p = a_p.item()
        a_p_vec = np.append(a_p_vec,a_p)

        phi_p,d_phi_p = phi(func,x_k,a_p,p_k, mu, Lambda) #calculating function values at a_p

        phi_p_vec = np.append(phi_p_vec,phi_p)

        #if the new point is above the sufficient decrease line or higher than the low point
        if (phi_p > phi_0 + mu_1*a_p*d_phi_0) or (phi_p > phi_low):
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

        if k >20:
            #print("Pinpoint iteration more than 20. Most likely stuck")
            return a_p, phi_p, a_p_vec, phi_p_vec


def Quad_Interp(a_high, a_low, phi_high, phi_low, d_phi_low):
    numerator = 2 * a_low * (phi_high - phi_low) + d_phi_low * (a_low**2 - a_high**2)
    denominator = 2 * (phi_high - phi_low + d_phi_low * (a_low - a_high))
    
    return numerator / denominator   
