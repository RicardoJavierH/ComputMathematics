import numpy as np
import biblioteca as bib
import scipy.sparse as sp
import scipy.sparse.linalg as spla
np.set_printoptions(precision=4,suppress=True)

A = [[1,2,3,4,5],[2,2,0,0,0],[3,0,3,0,0],[4,0,0,4,0],[5,0,0,0,5]]
"""
A = [[1,0,0,0,0,6,0,0,0,10,0],
     [0,2,0,4,0,0,0,0,2,0,0],
     [0,0,3,0,5,0,3,0,0,0,11],
     [0,4,0,4,0,0,7,0,0,0,0],
     [0,0,5,0,5,0,0,8,0,0,5],
     [6,0,0,0,0,6,0,0,6,0,0],
     [0,0,3,7,0,0,6,0,0,0,7],
     [0,0,0,0,8,0,0,8,0,0,8],
     [0,2,0,0,0,6,0,0,9,0,9],
     [10,0,0,0,0,0,0,0,0,10,10],
     [0,0,11,0,5,0,7,8,9,10,11]]
"""
A2 = np.array(A)
#print("A2-A2.transpose() =\n",A2-A2.T)

''' Using NumPy '''
Mat = np.array(A)
print(Mat)

LU = bib.LUdescomp(Mat)
L = LU[0]
U = LU[1]

print("L=\n",L)
print("U=\n",U)

"""
''' Using SciPy '''
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.SuperLU.html
spMat = sp.csc_array(A)
print(type(spMat))
lu = spla.splu(spMat,'NATURAL')

spL = lu.L.A # numpy.ndarray
spU = lu.U.A # numpy.ndarray

print("spL=\n",spL)
print("spU=\n",spU)

Pr = sp.csc_matrix((np.ones(5), (lu.perm_r, np.arange(5))))
Pc = sp.csc_matrix((np.ones(5), (np.arange(5), lu.perm_c)))

print(Pr.T @ (spL @ spU) @ Pc.T)
"""