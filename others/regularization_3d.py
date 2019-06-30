import numpy as np
import numpy as np
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

m = 2 #dimension
mean = np.array([8, 8])
sigma = 50 * np.eye(m)
N = 1000
x1 = np.linspace(-10, 10, N)
x2 = np.linspace(-10, 10, N)

X1, X2 = np.meshgrid(x1, x2)
X = np.c_[np.ravel(X1), np.ravel(X2)]

Y_plot = multivariate_normal.pdf(x=X, mean=mean, cov=sigma)
Y_plot = Y_plot.reshape(X1.shape) * -2000

fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection='3d')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

# draw surf
ax.plot_surface(X1, X2, Y_plot, cmap='bwr', linewidth=0, alpha=0.5)
cset = ax.contour(X1, X2, Y_plot, zdir='z', offset=-10, cmap=cm.coolwarm)

# draw sphere
u = np.linspace(0, np.pi, 30)
v = np.linspace(0, 2 * np.pi, 30)
r = 5

x = r * np.outer(np.sin(u), np.sin(v))
y = r * np.outer(np.sin(u), np.cos(v))
z = r * np.outer(np.cos(u), np.ones_like(v))
ax.plot_surface(x, y, z, color='g', alpha=0.5)
cset = ax.contour(x, y, z, zdir='z', levels=[0], offset=-10, cmap=cm.coolwarm)

xline=((-10, 10),(0,0),(0,0))
ax.plot(xline[0],xline[1],xline[2],'k')
yline=((0,0),(-10, 10),(0,0))
ax.plot(yline[0],yline[1],yline[2],'k')
zline=((0,0),(0,0),(-10,10))
ax.plot(zline[0],zline[1],zline[2],'k')
plt.show()
