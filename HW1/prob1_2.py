#Importing associated libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

# Generate x1, x2, and f values
x_1 = np.linspace(-50,50,1000)
x_2 = x_1
x1,x2 = np.meshgrid(x_1,x_2)
f = x1**3 + 2*x1*x2**2 - x2**3 - 20*x1

#Starting figure
fig  = plt.figure()

#Note that the contour plot is commented out here. This code plots both contourf and surface plot for ease of visualization
 # contour = plt.contourf(x1, x2, f, levels=50, cmap='viridis')

ax = fig.add_subplot(111, projection='3d')

#Labeling graphic minimums
circle = ax.scatter(0,0,0,s = 50, zorder = 2, fc = 'red',label = "Minimum")
surf = ax.plot_surface(x1, x2, f, cmap='viridis')

plt.colorbar(surf)
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()
