import numpy as np
import numpy.linalg as la

def gradConj1(A,b,x0,tol):
    r0 = A @ x0 - b
    p0 = -r0
    while(la.norm(r0) > tol):
        alpha = -(r0.T @ p0)/(p0.T @ A @ p0)
        x1 = x0 + alpha * p0
        r1 = A @ x1 - b
        beta = (r1.T @ A @ p0)/(p0.T @ A @ p0)
        p1 = -r1 + beta * p0

        x0 = x1
        r0 = r1
        p0 = p1

    return x0

"Programa principal"
A = np.random.randint(-10,11,(3,3))
A = A @ A.T
print(A)
b = np.random.randint(-10,11,(3,1))
print(b)
x0 = np.zeros_like(b)
resp = gradConj1(A,b,x0,0.01)
print(resp)
print(A @ resp - b)

resp2 = la.solve(A,b)
print(resp2)
print(A @ resp2 - b)