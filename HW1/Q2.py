import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Generate x and y values
x_1 = np.linspace(-100,100,1000)
x_2 = np.linspace(-100,100,1000)

x1,x2 = np.meshgrid(x_1,x_2)

f = x1**3 + 2*x1*x2**2 - x2**3 - 20*x1

plt.figure
plt.contourf(x1,x2,f,levels = 20)
plt.colorbar
plt.show()


