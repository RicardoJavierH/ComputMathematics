import numpy as np
import biblioteca as bib
import matplotlib.pyplot as plt
import numpy.polynomial as P


''' Interpolación '''
#x=np.array([1.,1.3,1.6,1.9,2.2])
#y=np.array([0.7651977,0.6200860,0.4554022,0.2818186,0.1103623])

x = np.linspace(-4,4,8)
y = np.arctan(x)

pol=bib.interpNewton(x,y) #interpolación de Newton por diferencias divididas
#pol=bib.interpLagrange(x,y)
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

