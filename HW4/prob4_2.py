import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define the variables
b = 0.125 #meter
h = 0.250 #meter 
P = 100E3 #Newtons
l = 1 #meter
sigma_yield = 200E6 #Pascal
tau_yield = 116E6 #Pascal
t_w,t_b,s1,s2,sigma_1,sigma_2 = sp.symbols('t_w t_b s1 s2 sigma_1 sigma_2')

# Define the equation
I = (h**3 / 12) * t_w + (b / 6) * t_b**3 + (h**2 * b / 2) * t_b
L_equation = 2*b*t_b + h*t_w + sigma_1 * ((P*l*h)/(2*I) - sigma_yield + s1**2) + sigma_2 * ((1.5*P)/(h*t_w) - tau_yield + s2**2)

#Calculate first order optimality

DL_Dtb = sp.diff(L_equation,t_b)
DL_Dtw = sp.diff(L_equation,t_w)
DL_Dsigma_1 = sp.diff(L_equation,sigma_1)
DL_Dsigma_2 = sp.diff(L_equation,sigma_2)
DL_Ds1  = sp.diff(L_equation,s1)
DL_Ds2  = sp.diff(L_equation,s2)

print(DL_Dtb)
print(DL_Dtw)
print(DL_Dsigma_1)
print(DL_Dsigma_2)
print(DL_Ds1)
print(DL_Ds2)


#Both Constraints Active
substitutions = {
    s1: 0,
    s2: 0,
}
solution_correct = sp.solve((DL_Dtb.subs(substitutions),DL_Dtw.subs(substitutions),DL_Dsigma_1.subs(substitutions),DL_Dsigma_2.subs(substitutions), DL_Ds1.subs(substitutions),DL_Ds2.subs(substitutions)),(t_w,t_b,s1,s2,sigma_1,sigma_2))
print(solution_correct)

#Both Constraints Inactive

substitutions = {
    sigma_1: 0,
    sigma_2: 0,
}
solution = sp.solve((DL_Dtb.subs(substitutions),DL_Dtw.subs(substitutions),DL_Dsigma_1.subs(substitutions),DL_Dsigma_2.subs(substitutions), DL_Ds1.subs(substitutions),DL_Ds2.subs(substitutions)),(t_w,t_b,s1,s2,sigma_1,sigma_2))
print(solution)

substitutions = {
    s1: 0,
    sigma_2: 0,
}
solution = sp.solve((DL_Dtb.subs(substitutions),DL_Dtw.subs(substitutions),DL_Dsigma_1.subs(substitutions),DL_Dsigma_2.subs(substitutions), DL_Ds1.subs(substitutions),DL_Ds2.subs(substitutions)),(t_w,t_b,s1,s2,sigma_1,sigma_2))
print(solution)

substitutions = {
    sigma_1: 0,
    s2: 0,
}
solution = sp.solve((DL_Dtb.subs(substitutions),DL_Dtw.subs(substitutions),DL_Dsigma_1.subs(substitutions),DL_Dsigma_2.subs(substitutions), DL_Ds1.subs(substitutions),DL_Ds2.subs(substitutions)),(t_w,t_b,s1,s2,sigma_1,sigma_2))
print(solution)



#Plotting
# Declare the values for the unknown variables (You can fill these in)

# Create a grid of t_b and t_w values
t_b = np.linspace(1E-4, 0.1, 400)  # Assuming a range [0,10] for t_b, change as needed
t_w = np.linspace(1E-4, 0.1, 400)  # Assuming a range [0,10] for t_w, change as needed
T_b, T_w = np.meshgrid(t_b, t_w)

# Calculate the objective function
F = 2*b*T_b + h*T_w

I = (h**3 / 12) * T_w + (b / 6) * T_b**3 + (h**2 * b / 2) * T_b

# Constraints
constraint1 = P * l * h / (2 * I) - sigma_yield
constraint2 = 1.5 * P / (h * T_w) - tau_yield

# Plot
plt.contourf(T_b, T_w, F, 50, cmap='viridis')  # '50' denotes number of contour levels
plt.contourf(T_b, T_w, constraint1, levels=[0, constraint1.max()], colors=['blue'], alpha=0.5)
legend_line_1 = Line2D([0], [0], color='blue', label='Constraint 1', linewidth=10)

plt.contourf(T_b, T_w, constraint2, levels=[0, constraint2.max()], colors=['green'], alpha=0.5)
legend_line_2 = Line2D([0], [0], color='green', label='Constraint 2', linewidth=10)

x_optimal = plt.scatter(0.01426039,0.0051724,label = "x*",zorder = 2, color = 'red')

"""
plt.xlabel('t_b')
plt.ylabel('t_w')
plt.legend(handles=[legend_line_1, legend_line_2, x_optimal])
plt.title('Objective Function Contour Plot')
plt.grid(True)
plt.colorbar()
plt.show()
"""