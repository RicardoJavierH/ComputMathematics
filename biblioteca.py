from turtle import shape
import numpy as np
import numpy.polynomial as P
import scipy.linalg as la

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
        

def sustRegresiva(A,b):   #Resuelve un sistema escalonado
    N = b.shape[0] # A y b deben ser array numpy bidimensional
    x = np.zeros((N,1))
    for i in range(N-1,-1,-1):
        x[i,0] = (b[i,0]-np.dot(A[i,i+1:N],x[i+1:N,0]))/A[i,i]
    return x # Array bidimensional

def sustProgresiva(A,b):   #Resuelve un sistema escalonado
    N = b.shape[0] # A y b deben ser array numpy bidimensional
    x = np.zeros((N,1))
    for i in range(0,N):
        x[i,0] = (b[i,0]-np.dot(A[i,0:i],x[0:i,0]))/A[i,i]
    return x # Array bidimensional

def GaussElimSimple(A,b):
    Ab = np.append(A,b,axis=1)
    escalonaSimple(Ab)
    A1 = Ab[:,0:Ab.shape[1]-1].copy()
    b1 = Ab[:,Ab.shape[1]-1].copy()
    b1 = b1.reshape(b.shape[0],1)
    x = sustRegresiva(A1,b1)
    return x # Array bidimensional

def GaussElimPiv(A,b):
    Ab = np.append(A,b,axis=1)
    escalonaConPiv(Ab)
    A1 = Ab[:,0:Ab.shape[1]-1].copy()
    b1 = Ab[:,Ab.shape[1]-1].copy()
    b1 = b1.reshape(b.shape[0],1)
    x = sustRegresiva(A1,b1)
    return x # Array bidimensional

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

def metNewtonSistNoLin(fun,jacfun,solAprox,nIter): #solAprox is ndarray
    solAprox = np.array(solAprox)
    print(solAprox)
    for i in range(nIter):
        A = np.array(jacfun(solAprox))
        b = np.array(fun(solAprox))
        b = b.reshape(2,1)
        Y = GaussJordanPiv(A,b)
        Y = np.reshape(Y,2)
        solAprox = solAprox - Y

    return solAprox 

def interpLagrange(cx,cy):
  n = len(cx)
  p = P.Polynomial([0])
  for i in range(n):
    mascara = np.ones(n,dtype=bool)
    mascara[i] = False
    raices = cx[mascara]
    Laux = P.Polynomial.fromroots(raices)
    p = p + cy[i]*Laux/Laux(cx[i])
  return p


def interpSisteLin(x,y):
    n = len(x)
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i,j] = np.power(x[i],j)

    b = np.zeros((n,1))
    for i in range(n):
        b[i,0] = y[i]

    vect = la.solve(A,b)
    vect.resize(n)
    pol = P.Polynomial(vect)
    return pol

def EulerODE(f,y0,t):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1,n):
        h = t[i] - t[i-1]
        y[i] = y[i-1] + h * f(y[i-1],t[i-1])
    return y

def RK4(f,y0,t):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(1,n):
        h = t[i] - t[i-1]
        k1 = f(y[i-1],t[i-1])*h
        k2 = f(y[i-1]+k1/2,t[i-1]+h/2)*h
        k3 = f(y[i-1]+k2/2,t[i-1]+h/2)*h
        k4 = f(y[i-1]+k3,t[i-1]+h)*h
        y[i] = y[i-1] + (k1+2*k2+2*k3+k4)/6
    return y

def RK4Systems(f,xy0,t):
    n = len(t)
    x = np.zeros(n)
    y = np.zeros(n)
    
    x0, y0 = xy0
    x[0] = x0
    y[0] = y0
    for i in range(1,n):
        h = t[i] - t[i-1]
        kx1, ky1 = f([x[i-1],y[i-1]],t[i-1])
        kx2, ky2 = f([x[i-1]+kx1*h/2,y[i-1]+ky1*h/2],t[i-1]+h/2)
        kx3, ky3 = f([x[i-1]+kx2*h/2,y[i-1]+ky2*h/2],t[i-1]+h/2)
        kx4, ky4 = f([x[i-1]+kx3*h,y[i-1]+ky3*h],t[i-1]+h)
        
        x[i] = x[i-1] + h*(kx1+2*kx2+2*kx3+kx4)/6
        y[i] = y[i-1] + h*(ky1+2*ky2+2*ky3+ky4)/6
        xy_t = np.array([x,y]).T
    return xy_t

def RK4SystemasODE(f, xy0, t):
    n = len(t)
    x = np.zeros(n)
    y = np.zeros(n)
    x[0], y[0] = xy0

    for i in range(1,n):
        h = t[i] - t[i-1]
        kx1, ky1 = f([x[i-1],y[i-1]],t[i-1])

        kx2, ky2 = f([x[i-1]+kx1*h/2,y[i-1]+ky1*h/2],t[i-1]+h/2)
        kx3, ky3 = f([x[i-1]+kx2*h/2,y[i-1]+ky2*h/2],t[i-1]+h/2)
        kx4, ky4 = f([x[i-1]+kx3*h,y[i-1]+ky3*h],t[i-1]+h)

        x[i] = x[i-1] + h*(kx1+2*kx2+2*kx3+kx4)/6
        y[i] = y[i-1] + h*(ky1+2*ky2+2*ky3+ky4)/6

        xy_t = np.array([x,y]).T

    return xy_t