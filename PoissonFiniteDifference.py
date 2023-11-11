import numpy as np
import matplotlib.pyplot as plt

''' Datos del problema'''
# Omega = [a,b]x[a,b]
a = 0
b = 1
N = 8 # Número de puntos de discretización en cada eje

Pi = np.pi # Constante pi de numpy

def u(x,y): #solución exacta del problema de Poisson
    return np.sin(Pi*x)*np.sin(Pi*y)

def f(x,y): # Función fuente
    ans = 2*np.power(Pi,2)*np.sin(Pi*x)*np.sin(Pi*y)
    return ans

def g(x,y): # Condición de frontera
    return 0

'''Discretización del dominio omega'''
x = np.linspace(a,b,N)
y = np.linspace(a,b,N)
h = (b-a)/(N-1)

X, Y = np.meshgrid(x, y)
fig = plt.figure()
plt.plot(x[1],y[1],'ro',label='u no conocido');
plt.plot(X,Y,'ro');
plt.plot(np.ones(N),y,'bo',label='Condición de frontera');
plt.plot(np.zeros(N),y,'bo');
plt.plot(x,np.zeros(N),'bo');
plt.plot(x, np.ones(N),'bo');
plt.xlim((-0.1,1.1))
plt.ylim((-0.1,1.1))
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(r' Dominio discretizado $\Omega_h,$ N= %s'%(N),fontsize=16,y=1.08)
plt.show();

'''Ensamblaje del sistema de ecuaciones'''


'''Resolución del sistema'''

'''Gráfica de la solución aproximada'''