#Importing associated libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

#Using a function to define equation
def equation(params):
    x,y = params
    return (1-x)**2 + (1-y)**2 + 0.5*(2*y-x**2)**2

#Callback function that is used to trace path in the minimize function
path = []
def callback (params):
    path.append(np.copy(params))

#Initial starting point
initial_guess = [100, 100]
result = minimize(equation, initial_guess, callback = callback)

#Printing results to ensure optimization success
print(result)

#Converting path array to numpy array.
path = np.array(path)

#Evaluate the path coordinates
path_eval = equation([path[:,0],path[:,1]])

x = np.linspace(-(np.max(path)+2), np.max(path)+2, 400)
#x = np.linspace(-2,2,1000)
y = x
X, Y = np.meshgrid(x, y)
Z = equation([X, Y])

#Plotting contour and the optimization path
plt.figure()
plt.plot(path[:,0],path[:,1],'-o',label = 'Optimization Path')
plt.contourf(X, Y, Z, levels = 100, cmap='jet')
plt.scatter(path[0,0],path[0,1],s = 50,fc = 'red',zorder = 2,label = 'Optimization Starting Point')
plt.scatter(path[-1,0],path[-1,1],s = 50,fc = 'green',zorder = 2,label = 'Optimization End Point')

plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

#Plotting iteration vs. path evaluation
plt.figure()
plt.plot(path_eval,'-o',label = 'Path Evaluations')
plt.axhline(y=0.091943, color='r', linestyle='--',label = 'Minimum = 0.091943')
plt.xlabel('Iteration')
plt.ylabel('f(x_1,x_2)')
plt.legend()
plt.show()


