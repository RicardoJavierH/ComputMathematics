import numpy as np
import biblioteca as bib
from scipy import integrate 
import matplotlib.pyplot as plt

a, b, c, d = 0.4, 0.002, 0.001, 0.7
def f(xy, t): # x: Nro de presas, y: Nro de depredadores 
    x, y = xy
    return [a * x - b * x * y, c * x * y - d * y]
# a: tasa de crecimiento de las presas
# b: habilidad de los depredadores para atrapar a las presas
# c: habilidad de las presas para escapar de los depredadores
# d: tasa de mortalidad de los depredadores

T = 100 # tiempo final
xy0 = [600, 40]
t = np.linspace(0, T, 250)
xy_t = integrate.odeint(f, xy0, t)
xy_t.shape

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].plot(t, xy_t[:,0], 'r', label="Prey")
axes[0].plot(t, xy_t[:,1], 'b', label="Predator")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Number of animals")
axes[0].legend()

axes[1].plot(xy_t[:,0], xy_t[:,1], 'k')
axes[1].set_xlabel("Number of prey")
axes[1].set_ylabel("Number of predators")

plt.show()
