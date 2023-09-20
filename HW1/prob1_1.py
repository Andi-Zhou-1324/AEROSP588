#Importing associated library
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Generate x and y values. y in this case is used as a subsitution for f for ease of understanding
x = np.linspace(-30,20,1000)
y = (1/12)*x**4 + x**3 - 16*x**2 + 4*x + 12

# Finding the location of the global minimum value by using the minimum command-
y_min_indx = np.argmin(y)

# Label for the equation
equation_label =  r'$y = \frac{1}{12}x^4 + x^3 - 16x^2 + 4x + 12$'

plt.figure
plt.plot(x,y,label = equation_label)

#Using the scatter plot command to graphically label the local and global minimum
plt.scatter(x[y_min_indx],y[y_min_indx],color = 'red',label="P1",zorder = 2)
plt.scatter(6.8,-200,color = 'red',label="P2",zorder = 2)

#Plotting commands
plt.title("Q1")
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')

plt.show()