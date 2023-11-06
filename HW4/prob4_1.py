import numpy as np
import sympy as sym

x1 = sym.Symbol('x1')
x2 = sym.Symbol('x2')
Lambda = sym.Symbol('Lambda')
c  = sym.Symbol('c')

eq1 = -x2 + 2*Lambda
eq2 = -x1 + 2*Lambda
eq3 = 2*x1 + 2*x2 - c

solution = sym.solve((eq1,eq2,eq3),(x1,x2,Lambda))

f = -x1*x2
h = 2*x1 + 2*x2 - c

L = f + Lambda*h

H = np.array([[sym.diff(sym.diff(L,x1),x1), sym.diff(sym.diff(L,x1),x2)],
              [sym.diff(sym.diff(L,x2),x1), sym.diff(sym.diff(L,x2),x2)]])

print("The solution is")
print(solution)

#------------------------------------------------
A, Lambda, x_1, x_2 = sym.symbols('A, Lambda, x_1, x_2')

f = -(2*x_1 + 2*x_2)
h = x_1*x_2 - A

L = f + Lambda*h
#print(L)
eq1 = sym.diff(L,x_1)
#print(eq1)
eq2 = sym.diff(L,x_2)
#print(eq2)
eq3 = sym.diff(L,Lambda)
#print(eq3)

solution = sym.solve((eq1,eq2,eq3),(x_1,x_2,Lambda))
#print(solution)

#print(sym.diff(h,x_1), sym.diff(h,x_2))

H = np.array([[sym.diff(sym.diff(L,x_1),x_1), sym.diff(sym.diff(L,x_1),x_2)],
              [sym.diff(sym.diff(L,x_2),x_1), sym.diff(sym.diff(L,x_2),x_2)]])
#print(H)