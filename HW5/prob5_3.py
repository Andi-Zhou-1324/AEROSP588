import numpy as np
import matplotlib.pyplot as plt
from truss import tenbartruss

A = np.ones(10,dtype = "complex_")*0.01
h = 1E-8

FD_result = tenbartruss(A, h, grad_method='FD', aggregate=False)
print(FD_result[2])


h = 1E-200
CS_result = tenbartruss(A, h, grad_method='CS', aggregate=False)
print(CS_result[2])

h = 1E-200
DT_result = tenbartruss(A, h, grad_method='DT', aggregate=False)
print(DT_result[2])

h = 1E-200
AJ_result = tenbartruss(A, h, grad_method='AJ', aggregate=False)
print(AJ_result[2])

norm_diff = np.linalg.norm(FD_result[2] - AJ_result[2], 'fro')

print("norm is " + str(norm_diff))