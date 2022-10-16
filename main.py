import numpy as np
import sympy
import biblioteca as bib
np.set_printoptions(precision=4,suppress=True)

""" Programa principal """
x, y = sympy.symbols("x, y")
f_mat = sympy.Matrix([y - x**3 -2*x**2 + 1, y + x**2 - 1])
matJacob = f_mat.jacobian(sympy.Matrix([x, y]))
print(matJacob)


def f(x):
    return [x[1] - x[0]**3 - 2 * x[0]**2 + 1, x[1] + x[0]**2 - 1]

def f_jacobian(x):
    return [[-3*x[0]**2-4*x[0], 1], [2*x[0], 1]]


solAprox = bib.metNewtonSistNoLin(f, f_jacobian, [-3,2], 10)
print(solAprox)

'''
# B=np.array([[2,-3,5],[6,-1,3],[-4,1,-2]]) #matrix to analyse

A = np.random.uniform(-10,10,(4,4))
print(A)
b = np.random.uniform(-10,10,(4,1))
print(b)

print("Método Gauss simple: ")
x = bib.GaussElimSimple(A,b)
print(x)
print(A@x-b)
print("\n")

print("Método de Gauss con pivoteamiento parcial: ")
x = bib.GaussElimPiv(A,b)
print(x)
print(A@x-b)
print("\n")

print("Método de Gauss-jordan con pivoteamiento parcial: ")
x = bib.GaussJordanPiv(A,b)
print(x)
print(A@x-b)
print("\n")

print("LU:")
lu = bib.LUdescomp(A)
L = lu[0]
U = lu[1]
print(L)
print(U)
print(L@U)
print(A)
'''