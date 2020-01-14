import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt

def loadData():
    A = sio.loadmat('A{}.mat'.format(number))['A{}'.format(number)]
    x0 = sio.loadmat('b{}.mat'.format(number))['b{}'.format(number)]
    return A,b

def l1_norm(x):
    return np.sum(np.absolute(x))

def l2_norm(x):
    return np.sqrt(np.sum(np.power(x,2)))

def cost(x):
    res = t_*np.dot(c.T,x)
    for i in range(A.shape[0]):
        res -= np.log(b[i] - np.dot(A[i],x))
    return res.flatten()

def grad_c(x):
    res = 0
    for i in range(A.shape[0]):
        res += A[i] * 1 / (b[i] - np.dot(A[i],x))
    # print(res)
    return res.flatten() + t_*c.flatten()

def linesearch():
    stopping_crit = 1e-3
    x = x0.flatten()
    alpha, beta = 0.25, 0.45
    iteration = 0
    # print(grad_c(x))
    # input()
    while l2_norm(grad_c(x)) >= stopping_crit and iteration < 5:
        t = 1
        while cost(x-t*grad_c(x)) >= cost(x) - alpha*t*np.dot(grad_c(x).T,grad_c(x)):
            t = beta*t
            print('t: {}'.format(t))
        x = x - t*grad_c(x)
        if iteration%1 == 0:
            print('Objective function: {}'.format(cost(x)))
            print('l2 norm grad: {}'.format(l2_norm(grad_c(x))))
        iteration += 1
        
        # print(x)
    print('Objective function: {}'.format(cost(x)))
    print('l2 norm grad: {}'.format(l2_norm(grad_c(x))))



A_,b_ = loadData(3)
A = np.block([[-A_, np.eye(A_.shape[0])],
              [A_, np.eye(A_.shape[0])]])
b = np.block([[-b_],[b_]])
c = np.block([[np.zeros((A_.shape[1],1))],[np.ones((A_.shape[0],1))]]).flatten()
A = -A
b= -b

x0 = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),b.flatten()-5*np.ones(A.shape[0]))
t_ = 1
# x_init = opt.linprog(c, A_ub=A, b_ub=b, options={'maxiter':100, 'disp':False}).x
# print(x_init)
linesearch()

# a = np.array([1,2,3])
# b = np.array([4,5,6])
# print(np.add(np.add(a,b),a.T))
