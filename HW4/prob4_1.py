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

mat = np.array([[0, -1],[-1,0]])

print(np.linalg.eig(mat))
