import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Generate x and y values
x = np.linspace(-30,20,1000)
y = (1/12)*x**4 + x**3 - 16*x**2 + 4*x + 12

# Finding the location of the minimum value
y_min_indx = np.argmin(y)

equation_label =  r'$y = \frac{1}{12}x^4 + x^3 - 16x^2 + 4x + 12$'

plt.figure
plt.plot(x,y,label = equation_label)
plt.scatter(x[y_min_indx],y[y_min_indx],color = 'red',label=f'Min Point: ({x[y_min_indx]:.2f}, {y[y_min_indx]:.2f})')

min_point_label = ["Minimum Point: ",x[y_min_indx],y[y_min_indx]]
plt.title("Q1")
plt.legend()
plt.xlabel('x')
plt.ylabel('y')

plt.show()