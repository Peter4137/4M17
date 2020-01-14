import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt

def f(x):
    sum = 0
    for i in range(1,6):
        sum += i*np.sin((i+1)*x + i)
    return sum

def df_dx(x):
    sum = 0
    for i in range(1,6):
        sum += i*(i+1)*np.cos((i+1)*x + i)
    return sum

def d2f_dx2(x):
    sum = 0
    for i in range(1,6):
        sum += -i*(i+1)**2*np.sin((i+1)*x + i)
    return sum

def newton(x0,f,gradf, tolerance):
    x = x0
    while abs(f(x)) > tolerance:
        x = x - f(x)/gradf(x)
    return x

x0 = -1
tol = 1e-12
x_min = newton(x0, df_dx,d2f_dx2,tol)

print("Minimum of f(x)={} found at x={} to a tolerance of {}".format(f(x_min), x_min, tol))
x = np.linspace(-10,10,200)
y = f(x)
plt.plot(x,y, zorder=1)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.scatter(x_min, f(x_min), color='black', s=50, zorder=2)
plt.show()

# print(5*f(x_min))
