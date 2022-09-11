import numpy as np

def Escalona(A):
    dim=np.shape(A)
    nfil = dim[0]
    ncol = dim[1]
    for j in range(0,ncol):
        for i in range(j+1,nfil):
            ratio = A[i,j]/A[j,j]
            A[i,:] = A[i,:] - ratio * A[j,:]
    return A

A = np.random.normal(-10,11,(3,5))
#b = np.random.normal(-10,11,(5,5))
print(Escalona(A))
