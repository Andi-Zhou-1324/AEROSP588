import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

x = sym.Symbol('x')
f = sym.exp(x)/(sym.sqrt(sym.sin(x)**3 + sym.cos(x)**3))

df_dx = sym.diff(f,x)

df_dx_at_1_5 = df_dx.subs(x, 1.5).evalf()
print("The analytic derivative at x = 1.5 is " + str(df_dx_at_1_5))

def fun(x):
    out = np.exp(x)/(np.sqrt(np.sin(x)**3 + np.cos(x)**3))
    return out



#The analytic derivative for this function is:

#We code up the complex method first
def complexStep(x, h):
    x = 1.5 + 1j*h
    f = fun(x)
    df = np.imag(f)/(h)
    x = np.real(x)
    return df

complex_out = str(complexStep(1.5,1E-200))
print("The complex step derivative at x = 1.5 is " + str(complexStep(1.5,1E-200)))

#The Finite Difference Method
#Forward Difference

def ForwardDiff(x,h):
    return (fun(x+h) - fun(x))/h

def CentralDiff(x,h):
    return ((fun(x+h) - fun(x-h))/(2*h))



#Conducting error studies

N = 400

h = np.logspace(-1,-400,N)
x = 1.5
error = np.zeros((N,3))
k = 0
for i in h:
    out = np.array([ForwardDiff(x,i),CentralDiff(x,i),complexStep(x,i)])
    out = np.abs(out - df_dx_at_1_5)
    error[k,:] = out
    k += 1

plt.figure()
plt.plot(h, error[:,0], label = "Forward Difference")
plt.plot(h, error[:,1], label = "Backward Difference")
plt.plot(h, error[:,2], label = "Complex Step")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Step Size (h)")
plt.ylabel('Error')
plt.gca().invert_xaxis()
plt.legend()
plt.grid()
plt.show()
#--------------------------------------------------------------------------------------------------
#Automatic Differentiation:

#We initiate a class called dual numbers, containing both 
class DualNumber:
    def __init__(self, value, derivative):
        self.value = value
        self.derivative = derivative

    #We then proceed to overload the operators as follow
    def __add__(self, other):
        if isinstance(other, DualNumber):
            new_value = self.value + other.value
            new_derivative = self.derivative + other.derivative
        else:
            new_value = self.value + other
            new_derivative = self.derivative
        return DualNumber(new_value, new_derivative)

    def __sub__(self, other):
        if isinstance(other, DualNumber):
            new_value = self.value - other.value
            new_derivative = self.derivative - other.derivative
        else:
            new_value = self.value - other
            new_derivative = self.derivative
        return DualNumber(new_value, new_derivative)

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            new_value = self.value*other.value
            new_derivative = (other.derivative*self.value + other.value*self.derivative)
        else: 
            new_value = self.value*other
            new_derivative = self.derivative*other
        return DualNumber(new_value, new_derivative)

    def __truediv__(self,other):
        if isinstance(other, DualNumber):
            # If the other operand is also a DualNumber, apply the quotient rule
            new_value = self.value / other.value
            new_derivative = (self.derivative * other.value - self.value * other.derivative) / (other.value ** 2)
        else:
            # If the other operand is a constant (int or float), only the derivative needs to change
            new_value = self.value / other
            new_derivative = self.derivative / other
        return DualNumber(new_value, new_derivative)

    def __pow__(self, other):
        if isinstance(other, DualNumber):
            # The general power rule: (u^v)' = u^v * (v' * ln(u) + v * u' / u)
            new_value = self.value ** other.value
            new_derivative = (other.value * (self.value ** (other.value - 1)) * self.derivative + (self.value ** other.value) * np.log(self.value) * other.derivative)
            return DualNumber(new_value, new_derivative)
        else:
            # Assuming 'other' is a constant
            new_value = self.value ** other
            # Power rule: (u^n)' = n * u^(n-1) * u'
            new_derivative = other * (self.value ** (other - 1)) * self.derivative
            return DualNumber(new_value, new_derivative)
    
    #Adding overloading functions for sin, cos and exp:
    def exp(self):
        """Compute the exponential of a dual number."""
        value = np.exp(self.value)
        # The derivative of exp(x) is exp(x), so we multiply by the original derivative
        derivative = np.exp(self.value) * self.derivative
        return DualNumber(value, derivative)

    def sin(self):
        """Compute the sine of a dual number."""
        value = np.sin(self.value)
        # The derivative of sin(x) is cos(x), so we multiply by the original derivative
        derivative = np.cos(self.value) * self.derivative
        return DualNumber(value, derivative)

    def cos(self):
        """Compute the cosine of a dual number."""
        value = np.cos(self.value)
        # The derivative of cos(x) is -sin(x), so we multiply by the original derivative
        derivative = -np.sin(self.value) * self.derivative
        return DualNumber(value, derivative)

    def sqrt(self):
        # Calculate the square root of the value
        value_sqrt = np.sqrt(self.value)
        # The derivative of sqrt(x) is 1 / (2 * sqrt(x)), so we apply the chain rule
        derivative_sqrt = self.derivative / (2 * value_sqrt)
        return DualNumber(value_sqrt, derivative_sqrt)

#We are looking for derivative at x = 1.5. We then set d_x = 1
x = DualNumber(1.5,1)
y = DualNumber(2,1)

out = x.exp()/((x.sin()**3 + x.cos()**3).sqrt())

diff_complex_AD = complexStep(1.5,1E-200) - out.derivative

print(out.value, out.derivative)
