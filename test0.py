import numpy as np
from cmath import pi

A = np.array([1,2,3])
B = np.array([[1,2,3],[4,5,6]])
print("A= ",A )
print("B= ",B)
print(A.size)
print(B.shape)
print(B[0,1])

A1 = A[[0,1]]
B1 = B[[0],[1,2]]
B2 = B[[1],:]
print("A1= ", A1)
print("B1= ", B1)
print("B2= ", B2)

print(np.sin(pi/6))
print(np.sin(np.pi/6))


def f(x,y):
    return x+y

print(f(2.5,3.7))

arr = np.array([[0,2,4],[6,8,10]]) 
app_arr = np.append(arr, [13,15,17]) 


def cuadrado(x):
    return np.square(x)
