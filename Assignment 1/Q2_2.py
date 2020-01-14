import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt

def loadData(number):
    A = sio.loadmat('A{}.mat'.format(number))['A{}'.format(number)]
    b = sio.loadmat('b{}.mat'.format(number))['b{}'.format(number)]
    return A,b

def l2_norm(x):
    return np.sqrt(np.sum(np.power(x,2)))

def phi(x):
    result = 0
    for i in range(2*m):
        if b_[i]-np.dot(A_[i],x) < 0:
            return np.inf
        result -= np.log(b_[i]-np.dot(A_[i],x))
    return result

def f(x):
    return t*np.dot(c_,x) + phi(x)

def grad_f(x):
    result = t*c_
    for i in range(2*m):
        result += A_[i]/(b_[i]-np.dot(A_[i],x))
    return result


def gradient_descent(x0):
    x = x0
    dx = -grad_f(x)
    all_f0 = np.array([])
    iteration = 0
    while l2_norm(dx) > stopping_crit:
        dx = -grad_f(x)
        tau = 1
        while f(x+tau*dx) >= f(x) - alpha*tau*np.dot(dx, dx):
            tau = tau*beta
        x = x + tau*dx
        iteration += 1
        if iteration%10 == 0:
            print("Step: {}".format(iteration))
            print("Objective function value: {}".format(f(x)))
        all_f0 = np.append(all_f0, t*np.dot(c_,x))
    
    print("Final objective function value: {}".format(all_f0[-1]))
    plt.plot(all_f0-all_f0[-1])
    plt.ylabel("Error in minimisation")
    plt.xlabel("Iteration number")
    plt.xscale("log")
    plt.show()
    
A, b = loadData(3)
(m,n) = A.shape
A_ = np.block([[A,-np.eye(m)],[-A,-np.eye(m)]])
b_ = np.concatenate((b,-b), axis=None)
c_ = np.concatenate((np.zeros(n), np.ones(m)), axis=None)
x0 = np.dot(np.dot(np.linalg.inv(np.dot(A_.T,A_)),A_.T),b_-5*np.ones(2*m))

t = 1
stopping_crit = 1e-3
alpha = 0.25
beta = 0.45

gradient_descent(x0)
