#https://a-ziaeemehr.medium.com/numerical-solving-system-of-equations-ill-conditioned-9253ca92eae2
import numpy as np
from numpy.linalg import norm, cond, solve

def hilbert_matrix(n):
    A = np.zeros((n,n))
    for i in range(1,n+1):
        for j in range(1,n+1):
            A[i-1,j-1] = 1/(i+j-1)
    return A

print("{:10s} {:20s} {:30s}".format("n","cond","error"))
print("-"*50)

for i in range(4,20):
    x = np.ones(i)
    H = hilbert_matrix(i)
    b = H.dot(x)

    c = cond(H)
    xx = solve(H,b)
    err = norm(x-xx,np.inf)/norm(x,np.inf)
    #solutions.append(xx)

    print("{:2d} {:20e} {:20e}".format(i,c,err))