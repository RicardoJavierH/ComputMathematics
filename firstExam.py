import numpy as np
import biblioteca as bib
np.printoptions(precision=3,suppress=True)

''' Firts question '''
def matriz(n):
    A = np.zeros((n,n))
    for i in range(1,n+1):
        for j in range(1,n+1):
            A[i-1,j-1] = 1/(i+j-1)
    return A

print(matriz(5))

''' Second question '''
A = np.random.uniform(-10,10,(7,7))
print(A)
b = np.random.uniform(-10,10,(7,1))
print(b)

print("Método de Gauss-jordan con pivoteamiento parcial: ")
x = bib.GaussJordanPiv(A,b)
print("respuesta:\n",x)
res = A@x-b
print("residuo (Ax-b):\n",res)
print("\n")

print("Norma suma del residuo:\n",np.linalg.norm(res,1))

''' Third question '''
print("n    Nro de cond    error relativo     error residual")
print("-"*57)

for i in range(4,21):
    x_exact = np.ones(i)
    A = matriz(i)
    b = A.dot(x_exact)

    c = np.linalg.cond(A)
    x_aprox = np.linalg.solve(A,b)
    errResidual = np.linalg.norm(A.dot(x_aprox)-b,np.Inf)
    errRelativo = np.linalg.norm(x_exact-x_aprox,np.inf)/np.linalg.norm(x_exact,np.inf)
    #solutions.append(xx)

    print(i,"  ",c,"   ",errRelativo,"   ",errResidual)
