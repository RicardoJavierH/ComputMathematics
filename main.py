import numpy as np
import biblioteca as bib
np.set_printoptions(precision=4,suppress=True)

""" Programa principal """

A = np.random.uniform(-10,10,(4,4))
print(A)

B=np.array([[2,-3,5],[6,-1,3],[-4,1,-2]])

b = np.random.uniform(-10,10,(4,1))
print(b)

x = bib.GaussElimSimple(A,b)
print(x)

x = bib.GaussElimPiv(A,b)
print(x)

print(A@x-b)

lu = bib.LUdescomp(A)
L = lu[0]
U = lu[1]
print(L)
print(U)
print(L@U)
print(A)