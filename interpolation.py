import numpy as np
import biblioteca as bib
import matplotlib.pyplot as plt
import numpy.polynomial as P


''' Interpolación '''
#x=np.array([1,3,4,5])
#y=np.array([0,3,-1,-1])

x = np.linspace(-4,4,15)
y = np.arctan(x)

pol=bib.interpLagrange(x,y)
#pol=bib.interpSisteLin(x,y)

print(pol)
print(pol(x))

" Gráfica "

a = x.min()
b = x.max()
xx = np.linspace(a,b,100)
yy = pol(xx)

yexact = np.arctan(xx) 

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(xx,yexact,'g',lw=2,label='Solución exacta') 
ax.plot(xx,yy,'b',lw=2,label='Polinomio interpolante')
ax.plot(x,y,'ro',alpha=0.6,label='Datos')
ax.legend(loc=2)
ax.set_xlabel(r"$x$", fontsize=10)
ax.set_ylabel(r"$y$", fontsize=10)
plt.grid()
plt.show()

