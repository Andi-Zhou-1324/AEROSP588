import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from uncon_optimizer import Uncon_BFGS, uncon_optimizer
from matplotlib.lines import Line2D


# Define the variables
b = 0.125 #meter
h = 0.250 #meter 
P = 100E3 #Newtons
l = 1 #meter
sigma_yield = 200E6 #Pascal
tau_yield = 116E6 #Pascal
tw, tb, mu= sp.symbols('tw, tb, mu')

# Define the equation
I = (h**3 / 12) * tw + (b / 6) * tb**3 + (h**2 * b / 2) * tb
f = 2*b*tb + h*tw
g_1 = (P*l*h)/(2*I*sigma_yield) - 1
g_2 = (1.5*P)/(h*tw*tau_yield) - 1

f_hat = f + mu/2*(sp.Max(0,g_1)**2 + sp.Max(0,g_2)**2)

f_hat_tw_sym = sp.diff(f_hat,tw)
f_hat_tb_sym = sp.diff(f_hat,tb)

f_hat_Lambda_function = sp.lambdify((tw, tb, mu), f_hat, 'numpy')
f_hat_Lambda = np.vectorize(f_hat_Lambda_function)


f_hat_tw_Lambda = sp.lambdify((tw,tb,mu),f_hat_tw_sym,'numpy')
f_hat_tb_Lambda = sp.lambdify((tw,tb,mu),f_hat_tb_sym,'numpy')


#thickness = np.array([[tw],[tb]])
def f_hat_func (thickness,mu):
    tw = thickness[0,0]
    tb = thickness[1,0]
    f_hat_tw = f_hat_tw_Lambda(tw,tb,mu)
    f_hat_tb = f_hat_tb_Lambda(tw,tb,mu)

    f_hat    = f_hat_Lambda(tw,tb,mu)

    return f_hat, np.vstack([f_hat_tw,f_hat_tb])

mu = 0.5


def plot_contour(x_vec, mu_value):

    # Define the range of tb and th values for plotting
    tb_values = np.linspace(1E-4, 0.1, 400) # Adjust as needed
    tw_values = np.linspace(1E-4, 0.1, 400) # Adjust as needed
    TB, TW = np.meshgrid(tb_values, tw_values)
    
    Z = f_hat_Lambda(TW, TB, mu_value)

    F = 2*b*TB + h*TW

    I = (h**3 / 12) * TW + (b / 6) * TB**3 + (h**2 * b / 2) * TB

    # Constraints
    constraint1 = P * l * h / (2 * I) - sigma_yield
    constraint2 = 1.5 * P / (h * TW) - tau_yield

    # Plot
    plt.contourf(TB, TW, F, 50, cmap='viridis')  # '50' denotes number of contour levels
    plt.contourf(TB, TW, constraint1, levels=[0, constraint1.max()], colors=['blue'], alpha=0.5)
    legend_line_1 = Line2D([0], [0], color='blue', label='Constraint 1', linewidth=10)

    plt.contourf(TB, TW, constraint2, levels=[0, constraint2.max()], colors=['green'], alpha=0.5)
    legend_line_2 = Line2D([0], [0], color='green', label='Constraint 2', linewidth=10)

    plt.plot(x_vec[1,:],x_vec[0,:], color = 'cyan',label = 'Optimization Path')
    plt.scatter(x_vec[1,:],x_vec[0,:], color = 'cyan')
    plt.contour(TW, TB, Z, 100, colors = 'black')
    plt.colorbar()

    plt.xlabel('tb (t_b)')
    plt.ylabel('tw (t_h)')
    plt.title(f"mu = {mu}")
    plt.show()


#---------------------------Time to Optimize---------------------------------
diff = 10
rho  = 1.2
x    = np.array([[0.01],[0.1]])
k    = 0
x_vec = np.array(x)
while diff > 1E-8:
    x_past = x
    result = uncon_optimizer(f_hat_func, x, 1E-6, mu, options = None)

    x = result[0]
    mu = mu*rho
    diff = np.abs(np.max(x_past) - np.max(x))
    #print(diff)
    x_vec = np.hstack((x_vec,x))
    k = k + 1
 #   if k == 1 or k == 10 or k == 20:
 #       plot_contour(x_vec, mu)

#plot_contour(x_vec, mu)


#plt.plot(x_vec[0,:],x_vec[1,:], color = 'cyan',label = 'Optimization Path')
plt.show()
print(x)
print(mu)



"""

#--------------------------Interior Penalty Method----------------------------
# Define the variables
b = 0.125 #meter
h = 0.250 #meter 
P = 100E3 #Newtons
l = 1 #meter
sigma_yield = 200E6 #Pascal
tau_yield = 116E6 #Pascal
tw, tb, mu= sp.symbols('tw, tb, mu')

# Define the equation
I = (h**3 / 12) * tw + (b / 6) * tb**3 + (h**2 * b / 2) * tb
f = 2*b*tb + h*tw
g_1 = (P*l*h)/(2*I*sigma_yield) - 1
g_2 = (1.5*P)/(h*tw*tau_yield) - 1

f_hat = f - mu*(sp.log(-g_1) + sp.log(-g_2))

f_hat_tw_sym = sp.diff(f_hat,tw)
f_hat_tb_sym = sp.diff(f_hat,tb)

f_hat_Lambda_function = sp.lambdify((tw, tb, mu), f_hat, 'numpy')
f_hat_Lambda = np.vectorize(f_hat_Lambda_function)

f_hat_tw_Lambda = sp.lambdify((tw,tb,mu),f_hat_tw_sym,'numpy')
f_hat_tb_Lambda = sp.lambdify((tw,tb,mu),f_hat_tb_sym,'numpy')


#thickness = np.array([[tw],[tb]])
def f_hat_func (thickness,mu):
    tw = thickness[0,0]
    tb = thickness[1,0]
    f_hat_tw = f_hat_tw_Lambda(tw,tb,mu)
    f_hat_tb = f_hat_tb_Lambda(tw,tb,mu)

    f_hat    = f_hat_Lambda(tw,tb,mu)

    return f_hat, np.vstack([f_hat_tw,f_hat_tb])

def plot_contour(x_vec, mu_value):

    # Define the range of tb and th values for plotting
    tb_values = np.linspace(1E-4, np.max(x_vec[:]), 400) # Adjust as needed
    tw_values = np.linspace(1E-4, np.max(x_vec[:]), 400) # Adjust as needed
    TB, TW = np.meshgrid(tb_values, tw_values)
    
    Z = f_hat_Lambda(TW, TB, mu_value)

    F = 2*b*TB + h*TW

    I = (h**3 / 12) * TW + (b / 6) * TB**3 + (h**2 * b / 2) * TB

    # Constraints
    constraint1 = P * l * h / (2 * I) - sigma_yield
    constraint2 = 1.5 * P / (h * TW) - tau_yield

    # Plot
    plt.contourf(TB, TW, F, 50, cmap='viridis')  # '50' denotes number of contour levels
    plt.contourf(TB, TW, constraint1, levels=[0, constraint1.max()], colors=['blue'], alpha=0.5)
    legend_line_1 = Line2D([0], [0], color='blue', label='Constraint 1', linewidth=10)

    plt.contourf(TB, TW, constraint2, levels=[0, constraint2.max()], colors=['green'], alpha=0.5)
    legend_line_2 = Line2D([0], [0], color='green', label='Constraint 2', linewidth=10)

    plt.plot(x_vec[1,:],x_vec[0,:], color = 'cyan',label = 'Optimization Path')
    plt.scatter(x_vec[1,:],x_vec[0,:], color = 'cyan')
    plt.contour(TB, TW, Z, 100, colors = 'black')
    plt.colorbar()

    plt.xlabel('tb (t_b)')
    plt.ylabel('tw (t_h)')
    plt.title(f"mu = {mu}")
    plt.show()

mu = 0.1
#---------------------------Time to Optimize---------------------------------
diff = 10
rho  = 0.95
x    = np.array([[0.03],[0.008]])
k    = 0
x_vec = np.array(x)
while diff > 1E-6:
    x_past = x
    result = uncon_optimizer(f_hat_func, x, 1E-6, mu, options = None)

    x = result[0]
    mu = mu*rho
    diff = np.abs(np.max(x_past) - np.max(x))
    #print(diff,k)
    x_vec = np.hstack((x_vec,x))

    if k == 1 or k == 50 or k == 100:
        plot_contour(x_vec, mu)
    k = k + 1


plot_contour(x_vec, mu)
print(x)
print(mu)

"""