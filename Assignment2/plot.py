import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def schwefel(x):
    sum = 0
    for i in range(x.size):
        sum += -x[i]*np.sin(np.sqrt(abs(x[i])))
    return sum

x0 = np.linspace(-500, 500, 100)
x1 = np.linspace(-500, 500, 100)
[X0,X1] = np.meshgrid(x0,x1)
f = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        f[i][j] = schwefel(np.array([X0[i][j],X1[i][j]]))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X0, X1, f, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.show()