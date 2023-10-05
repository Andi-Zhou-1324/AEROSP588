import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
#Importing different libraries
plt.figure()
E_0_vec = np.array([0,10,1E2,1E3,1E4,1E5,1E6,1E7])
for i in E_0_vec:
    E_0 = i 

    e = 0.7
    M = np.pi/2

    ConCri = 1E-16
    E = E_0
    residual = 100

    residual_Newton = np.array([])
    E_k_Newton = np.array([])
    while np.abs(residual) >= ConCri:
        E_k = E - (E-e*np.sin(E) - M)/(1 - e*np.cos(E))    
        residual = np.abs(E_k - E)
        residual_Newton = np.append(residual_Newton, residual)
        E_k_Newton = np.append(E_k_Newton,E_k)
        E = E_k
    plt.plot(residual_Newton,label = "E = "+str(i))
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Residual")
plt.yscale('log')
plt.grid()
print ("For Newton Solver, E = "+str(E)+". We are able to strike down to 1E-16 of accuracy")
##----------------------------------------------------------------
plt.figure()

 #Dummy value for initialization
E_0_vec = np.array([0,10,1E2,1E3,1E4,1E5,1E6,1E7])
for i in E_0_vec:
    E = i
    residual_FixedPoint = np.array([])
    residual = 100
    while np.abs(residual) >= ConCri:
        E_k = M + e*np.sin(E)
        residual = np.abs(E_k - E)
        residual_FixedPoint = np.append(residual_FixedPoint, residual)
        E = E_k
    plt.plot(residual_FixedPoint,label = 'E = '+str(i))

plt.legend()
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Residual")
plt.grid()


plt.figure()
plt.plot(residual_Newton,label = "Newton's Method")
plt.plot(residual_FixedPoint, label = "Fixed Point Iteration")
plt.legend()
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Residual")
plt.grid()

print ("For Fixed_-Point Iteration Solver, E = "+str(E)+". We are able to strike down to 1E-16 of accuracy")



#-----------------------E vs M vs e---------------------------------------

plt.figure()
e_vec = np.array([0,0.1,0.5,0.9])
M_vec = np.linspace(0.0,2*np.pi,int(1E3))
for i in e_vec:
    E_vec = np.array([])
    for j in M_vec:
        E_0 = 10 #Default starting point

        e = i
        M = j

        ConCri = 1E-3
        E = E_0
        residual = 100

        residual_Newton = np.array([])
        E_k_Newton = np.array([])
        while np.abs(residual) >= ConCri:
            E_k = E - (E-e*np.sin(E) - M)/(1 - e*np.cos(E))    
            residual = np.abs(E_k - E)
            #print(residual)
            E = E_k
        E_vec = np.append(E_vec,E_k)
    plt.plot(M_vec,E_vec,label = "e = "+str(i))

plt.legend()
plt.xlabel("M")
plt.ylabel("E")
plt.grid()

#-------------------------Solver Noise-----------------------------------------------

plt.figure()
e_vec = np.array([0,0.1,0.5,0.9])
M_vec = np.linspace(np.pi-1E-6,np.pi+1E-6,int(1E4))

E_vec = np.array([])
for j in M_vec:
    E_0 = 10 + np.random.rand() #Default starting point with perturbation

    e = 0.7
    M = j

    ConCri = 1E-2
    E = E_0
    residual = 100
    while np.abs(residual) >= ConCri:
        E_k = E - (E-e*np.sin(E) - M)/(1 - e*np.cos(E))    
        residual = np.abs(E_k - E)
        #print(residual)
        E = E_k
    E_vec = np.append(E_vec,E_k)
plt.plot(M_vec,E_vec,label = "e = 0.7")

plt.legend()
plt.xlabel("M")
plt.ylabel("E")
plt.grid()
plt.title("Numerical Noise")
plt.show()