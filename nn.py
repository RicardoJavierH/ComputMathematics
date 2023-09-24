import scipy.sparse as sp
#import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt

#Matrices dispersas en formato COO
print("Formato COO")

values = [1, 2, 3, 4]
rows = [0, 1, 2, 3]
cols = [1, 3, 2, 0]

A = sp.coo_matrix((values, (rows, cols)), shape=[4, 4])
print(A)
print(A.data)


#Formato CSR
print("Formato CSR")

A.tocsr

print("valA = ",A.data)
print("IA = ",A.row)
print("JA = ",A.col)


N = 10
A = sp.eye(N, k=1) - 2 * sp.eye(N) + sp.eye(N, k=-1)
A = sp.diags([1, -2, 1], [1, 0, -1], shape=[N, N], format='csc')
fig, ax = plt.subplots()
ax.spy(A)