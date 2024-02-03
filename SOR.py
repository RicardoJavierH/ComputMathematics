import scipy.sparse as sp
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt
import time
import random as randint
from scipy.sparse import diags

# Construct the dense linear system
A = np.array([[4,3, 0],[3, 4, -1],[0, -1, 4]], dtype=float);
print("A=",A)
N=3
#res = np.random.rand(N)
#res = np.ones(4)
#res = np.array([1,1,1,1])
#b = A@res
b = np.array([24,30,-24], dtype=float);
print("b=",b)

x0 = np.array([1.,1.,1.],dtype=np.float64)
print("x0=",x0)

Dfull = np.diag(A)
print("Dfull=",Dfull)
D = np.diagflat(Dfull)
print("D=",D)

w = 1.25

Ll = np.tril(A, -1)
L = -1*np.array(Ll, dtype=float);
print("L=",L)

U=(-L+D)-A
print("U=",U)

DpluswL=D-w*L
print("DpluswL= \n",DpluswL)

DpluswLinv = np.zeros_like(DpluswL)
DpluswLinv = np.linalg.inv(DpluswL) # Mejorar el algoritmo de cálculo de inversa
print("DpluswLinv=",DpluswLinv)

wDU = (1-w)*D+w*U
G = DpluswLinv@wDU

inicio = time.time()

for cont in range(30): 
  x0 = G@x0 + w*DpluswLinv@b
  #print(cont,"\t",x0)
  
final = time.time()
tiempo_ejecucion = final - inicio
print("x0=",x0)
print(f"El tiempo de ejecución fue de {tiempo_ejecucion} segundos")

print("Aaprox-b: ",A@x0-b)

