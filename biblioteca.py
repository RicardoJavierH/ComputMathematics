import numpy as np

def operacionFila(A,fm,fp,factor): # filafm = filafm - factor*filafp
    A[fm,:] = A[fm,:] - factor*A[fp,:]

def intercambiaFil(A,fi,fj):
    A[[fi,fj],:] = A[[fj,fi],:]

def escalonaSimple(A):
    nfil = A.shape[0]
    ncol = A.shape[1]
    
    for j in range(0,nfil):
        for i in range(j+1,nfil):
            ratio = A[i,j]/A[j,j]
            operacionFila(A,i,j,ratio)


def escalonaConPiv(A):
    nfil = A.shape[0]
    ncol = A.shape[1]
    
    for j in range(0,nfil):
        imax = np.argmax(np.abs(A[j:nfil,j]))
        intercambiaFil(A,j+imax,j)
        for i in range(j+1,nfil):
            ratio = A[i,j]/A[j,j]
            operacionFila(A,i,j,ratio)

def reescalaFila(A,i,factor):
    A[i,:] = factor*A[i,:]
    return None

def escalonaReducidaConPiv(A): # convierte a la forma escalonada reducida
    nfil = A.shape[0]
    ncol = A.shape[1]
    
    for j in range(0,nfil):
        imax = np.argmax(np.abs(A[:,j]))
        intercambiaFil(A,imax,j)
        for i in range(0,nfil):
            if(i != j):
                ratio = A[i,j]/A[j,j]
                operacionFila(A,i,j,ratio)
        reescalaFila(A,j,1/A[j,j])
        

def sustRegresiva(A,b):   #resuelve un sistema escalonado
    N = b.shape[0]
    x = np.zeros((N,1))
    for i in range(N-1,-1,-1):
        x[i,0] = (b[i,0]-np.dot(A[i,i+1:N],x[i+1:N,0]))/A[i,i]
    return x

def GaussElimSimple(A,b):
    Ab = np.append(A,b,axis=1)
    escalonaSimple(Ab)
    A1 = Ab[:,0:Ab.shape[1]-1].copy()
    b1 = Ab[:,Ab.shape[1]-1].copy()
    b1 = b1.reshape(b.shape[0],1)
    x = sustRegresiva(A1,b1)
    return x

def GaussElimPiv(A,b):
    Ab = np.append(A,b,axis=1)
    escalonaConPiv(Ab)
    A1 = Ab[:,0:Ab.shape[1]-1].copy()
    b1 = Ab[:,Ab.shape[1]-1].copy()
    b1 = b1.reshape(b.shape[0],1)
    x = sustRegresiva(A1,b1)
    return x

def GaussJordanPiv(A,b):
    Ab = np.append(A,b,axis=1)
    escalonaReducidaConPiv(Ab)
    A1 = Ab[:,0:Ab.shape[1]-1].copy()
    b1 = Ab[:,Ab.shape[1]-1].copy()
    b1 = b1.reshape(b.shape[0],1)
    x = sustRegresiva(A1,b1)
    return x    

def LUdescomp(A): # A debe ser matriz cuadrada
    L = np.zeros_like(A)
    U = A.copy()

    nfil = A.shape[0]
    ncol = A.shape[1]
    
    for j in range(0,nfil):
          for i in range(j+1,nfil):
            ratio = U[i,j]/U[j,j]
            L[i,j] = ratio
            operacionFila(U,i,j,ratio)

    np.fill_diagonal(L,1)
    return (L,U)

def metNewtonSistNoLin(fun, jacfun, solAprox, nIter):
    solAprox = np.array(solAprox)
    for i in range(nIter):
        A = np.array(jacfun(solAprox))
        b = np.array(fun(solAprox))
        b = b.reshape(2,1)
        Y = GaussJordanPiv(A,b)
        Y = np.reshape(Y,2)
        solAprox = solAprox - Y
    
    return solAprox    



