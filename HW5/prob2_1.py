import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
#Importing different libraries
plt.figure()
E_0_vec = np.array([0])
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

print ("For Newton Solver, E = "+str(E)+". We are able to strike down to 1E-16 of accuracy")
##----------------------------------------------------------------