import numpy as np

def EchelonSimple(A):
    dim=np.shape(A)
    nfil = dim[0]
    ncol = dim[1]
    for j in range(0,ncol):
        for i in range(j+1,nfil):
            ratio = A[i,j]/A[j,j]
            A[i,:] = A[i,:] - ratio * A[j,:]
    return A

def echelonSimple(A):
    nf = np.shape(A)[0]
    nc = np.shape(A)[1]
    for j in range(nc-1):
        for i in range(j+1,nf):
            ratio = -A[i,j]/A[j,j]
            rowOperation(A,i,j,ratio)

def echelonRowPiv(A):            
    nf = np.shape(A)[0]
    nc = np.shape(A)[1]
    for j in range(nc-1):
        imax = np.argmax(np.abs(A[j:nf,j]))
        swapRows(A,j,j+imax)
        for i in range(j+1,nf):
            ratio = -A[i,j]/A[j,j]
            rowOperation(A,i,j,ratio)

def swapRows(M,fi,fj):
    M[[fi,fj]] = M[[fj,fi]]
    
def reescalingRow(A,fi,factor):
    A[fi,:] = factor*A[fi,:]

def rowOperation(M,fm,fp,factor):
    M[fm,:] += factor*M[fp,:]     

def backSubstitution(A,b):
    nf = np.shape(A)[0]
    nc = np.shape(A)[1]
    x = np.zeros((nf,1))
    for i in range(nc-1,-1,-1):
        x[i,0] = (b[i,0] - np.dot(A[i,i+1:nc],x[i+1:nc,0]))/A[i,i]
    return x

def GaussElimSimple(A,b):
    Ab = np.append(A,b,axis=1)
    echelonSimple(Ab)
    A2 = Ab[:,0:A.shape[1]].copy()
    b2 = Ab[:,Ab.shape[1]-1].copy()
    b2 = b2.reshape(5,1)
    x = backSubstitution(A2,b2)
    return x

def GaussElimRowPiv(A,b):
    Ab = np.append(A,b,axis=1)
    echelonRowPiv(Ab)
    A2 = Ab[:,0:A.shape[1]].copy()
    b2 = Ab[:,Ab.shape[1]-1].copy()
    b2 = b2.reshape(5,1)
    x = backSubstitution(A2,b2)
    return x


''' Main '''
np.set_printoptions(precision=5,suppress=True)
A = np.random.uniform(-10,11,(5,5))
b = np.random.normal(-10,11,(5,1))
print(A)
#echelonRowPiv(A)
x = GaussElimSimple(A,b)
print(x)
x = GaussElimRowPiv(A,b)
print(x)
print(A@x-b)