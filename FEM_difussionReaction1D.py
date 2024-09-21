import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import linalg
import bibliotecaFEM as bibfem

a = 0; b = 1 #Dominio del problema
#sigma = -np.power(np.pi,2)
sigma = 2.0
print(sigma)
crearNodosManual = False;
if (crearNodosManual):
    nodVec = np.array([0,0.3,0.7,1])
else:
    N=10 #Número de nodos
    nodVec=np.linspace(a,b,N) # generando nodos equiespaciados

print(nodVec)

def f(x):
    ans = -2*np.pi*np.cos(np.pi*x)+2*x*np.sin(np.pi*x)+np.power(np.pi,2)*x*np.sin(np.pi*x)
    return ans

def u(x):
  return x * np.sin(np.pi*x)

x=np.linspace(a,b,400)

#N: Número de elementos de la base
A = np.zeros((N-2,N-2),dtype=np.float64)
for i in range(N-2):
  for j in range(N-2):
    A[i,j]=integrate.quad(lambda x: bibfem.dphi(i+1,x,nodVec)*bibfem.dphi(j+1,x,nodVec),a,b)[0]

print(A)

B = np.zeros_like(A)
for i in range(N-2):
  for j in range(N-2):
    B[i,j]=integrate.quad(lambda x: bibfem.phi(i+1,x,nodVec)*bibfem.phi(j+1,x,nodVec),a,b)[0]

print(B)

F = np.zeros(N-2,dtype=np.float64)
for i in range(N-2):
  F[i] = integrate.quad(lambda x: f(x)*bibfem.phi(i+1,x,nodVec),a,b)[0]

print(F)

K = np.zeros_like(A)
K = A+sigma*B
print(K)
C = linalg.solve(K,F)
print(C)

def uh(x):
  ans = 0
  for i in range(N-2):
    ans += C[i]*bibfem.phi(i+1,x,nodVec)
  return ans

yh = uh(x)
yexact= u(x)
plt.plot(x,yh,'r',x,yexact,'b')
plt.grid()
plt.show()