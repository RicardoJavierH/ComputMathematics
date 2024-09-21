import numpy as np
import biblioteca as bib
np.set_printoptions(precision=4,suppress=True)


# B=np.array([[2,-3,5],[6,-1,3],[-4,1,-2]]) #matrix to analyse

A = np.random.uniform(-10,10,(4,4))
print(A)
b = np.random.uniform(-10,10,(4,1))
print(b)

'''
L=np.array([[1,0,0],[1,1,0],[1,1,1]])
b = np.array([[1],[2],[3]])
Y=bib.sustProgresiva(L,b)
print(Y)
print("Residuo:\n",L@Y-b)
'''
'''
print("Método Gauss simple: ")
x = bib.GaussElimSimple(A,b)
print("Solución aprox:\n",x)
print("Residuo:\n",A@x-b)
print("\n")
'''

'''
print("Método de Gauss con pivoteamiento parcial: ")
x = bib.GaussElimPiv(A,b)
print("Solución aprox:\n",x)
print("Residuo:\n",A@x-b)
print("\n")
'''

'''
print("Método de Gauss-jordan con pivoteamiento parcial: ")
x = bib.GaussJordanPiv(A,b)
print("Solución aprox:\n",x)
print("Residuo:\n",A@x-b)
print("\n")
'''

# AX=b <--> A=LU y LY=b (sust. prog.) y UX=Y (sust. reg.)
print("Solución por descomposición en LU:")
lu = bib.LUdescomp(A)
L = lu[0]
U = lu[1]
#print("L:\n",L)
#print("U:\n",U)
print("LU:\n",L@U)
print("A:\n",A)
Y=bib.sustProgresiva(L,b)
X=bib.sustRegresiva(U,Y)
print("Solución aprox:\n",X)
print("Residuo:\n",A@X-b)
print("\n")



