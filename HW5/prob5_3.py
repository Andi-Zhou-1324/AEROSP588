import numpy as np
import matplotlib.pyplot as plt
from truss import tenbartruss

A = np.ones(10,dtype = "complex_")
h = 1E-8

FD_result = tenbartruss(A, h, grad_method='FD', aggregate=False)
print(FD_result[2])


h = 1E-200
CS_result = tenbartruss(A, h, grad_method='CS', aggregate=False)
print(CS_result[2])
