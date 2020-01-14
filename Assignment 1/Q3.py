import numpy as np
import scipy.io as sio
import scipy.optimize as opt
from matplotlib import pyplot as plt
import random

def loadData():
    A = sio.loadmat('A.mat')['A']
    x0 = sio.loadmat('x0.mat')['x'].flatten()
    return A,x0

def problem(x):
    return l2_norm(np.dot(A,x)-b)**2 + l*l1_norm(x)

def l1_norm(x):
    return np.sum(np.absolute(x))

def l2_norm(x):
    return np.sqrt(np.sum(np.power(x,2)))

def inf_norm(x):
    return np.max(np.absolute(x))

def f(X):
    x = X[:n]
    u = X[n:]
    result = t*l2_norm(np.dot(A,x)-b)**2
    for i in range(n):
        result += t*l*u[i]
    return result + barrier(X)

def grad_f(X):
    x = X[:n]
    u = X[n:]
    g1 = 2*t*np.matmul(A.T, np.dot(A,x)-b)
    for i in range(n):
        g1[i] += (2*x[i])/(u[i]**2-x[i]**2) 
    g2 = t*l*np.ones(n)
    for i in range(n):
        g2[i] -= (2*u[i])/(u[i]**2-x[i]**2)
    return np.concatenate((g1, g2))

def hessian(X):
    x = X[:n]
    u = X[n:]
    h_11 = np.zeros(n)
    h_22 = np.zeros(n)
    for i in range(n):
        h_11[i] += (-4*x[i]*u[i])/(u[i]**2-x[i]**2)**2
        h_22[i] += (2*(u[i]**2+x[i]**2))/(u[i]**2-x[i]**2)**2
    H_12 = np.diag(h_11)
    H_22 = np.diag(h_22)
    H_11 = 2*t*np.dot(A.T,A) + H_22
    return np.block([[H_11, H_12],[H_12, H_22]])

def barrier(X):
    x = X[:n]
    u = X[n:]
    result = 0
    for i in range(n):
        if u[i] - x[i] < 0 or u[i] + x[i] < 0:
            return np.inf
        result -= np.log(u[i]-x[i]) + np.log(u[i]+x[i])
    return result

def G(v):
    return -0.25*np.dot(v,v) - np.dot(v,b)

def interior_point():
    iteration = 0
    x = np.concatenate((0.5*np.random.rand(n),np.ones(n)))
    dx = -np.dot(np.linalg.inv(hessian(x)),grad_f(x))

    while -np.dot(grad_f(x).T, dx)/2 > stopping_crit:
        tau = exact_linesearch(f,x,dx)      
        x = x+tau*dx
        dx = -np.dot(np.linalg.inv(hessian(x)),grad_f(x))
        
        print("Iteration number {}".format(iteration))
        print("phi(x,u): "+str(f(x)))
        print(problem(x[:n]))
        print(-np.dot(grad_f(x).T, dx)/2)
        print("-------------")
        
        iteration += 1
    return x
     
def exact_linesearch(f, x, dx):
    g = lambda tau: f(x+tau*dx)
    res = opt.minimize_scalar(g)
    return res.x

A,x0 = loadData()
# A = 2*(np.random.rand(60,256)-0.5)

(m,n) = A.shape

# num = 10
# index = np.random.randint(0,n,num)
# x0 = np.zeros(n)
# x0[index] = 1

b = np.dot(A,x0)
# print(np.linalg.cond(np.dot(A.T,A)))
# input()
# xmin = np.linalg.lstsq(A,b)[0]

# plt.plot(xmin)
# plt.grid(True)
# plt.xlabel('i')
# plt.ylabel('$x_i$')
# plt.ylim(-1,1)
# plt.show()


l_max = inf_norm(2*np.dot(A.T,b))
l = 0.01*l_max
t = 0.01
stopping_crit = 1e-10
best_x = interior_point()

print("Final minimised value: " + str(problem(best_x[:n])))

fig, axs = plt.subplots(3, 1)
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
axs[0].plot(x0)
axs[0].grid(True)
axs[0].set_xlabel('i')
axs[0].set_ylabel('$x_i$')
axs[0].set_title('Original signal')
axs[1].plot(best_x[:n]-np.linalg.lstsq(A,b)[0])
axs[1].grid(True)
axs[1].set_xlabel('i')
axs[1].set_ylabel('$x_i$')
axs[1].set_title('Reconstructed signal')
axs[2].plot(np.linalg.lstsq(A,b)[0])
axs[2].grid(True)
axs[2].set_xlabel('i')
axs[2].set_ylabel('$x_i$')
axs[2].set_title('Minimum energy reconstruction')

plt.show()




