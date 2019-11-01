import scipy.optimize as opt
import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt

#load in data from .mat files
A = sio.loadmat('A1.mat')['A1']
b = sio.loadmat('b1.mat')['b1']

c = 
Abar = np.array([[-A,1],[A,1]])
bbar = np.array([[-b],[b]])
result = opt.linprog(c, options={'A':Abar, 'b':bbar})

print(result)
print(np.dot(c.T, result.x))
