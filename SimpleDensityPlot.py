import pylab as plt
import numpy as np

# Sample data
side = np.linspace(-2,2,10)
X,Y = np.meshgrid(side,side)
Z = np.exp(-((X-1)**2+Y**2))

# Plot the density map using nearest-neighbor interpolation
plt.pcolormesh(X,Y,Z)


plt.show()