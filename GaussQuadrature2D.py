from scipy import integrate
from scipy import linalg, special # used for generate Gauss points
import numpy as np

def gauss_legendre(n, lower=-1, upper=1):
    '''
    Gauss-Legendre quadrature:
    
    A rule of order 2*n-1 on the interval [lower, upper] 
    with respect to the weight function w(x) = 1.
    '''
    nodes, weights = special.roots_legendre(n)
    if lower != -1 or upper != 1:
        nodes = (upper+lower)/2 + (upper-lower)/2*nodes
        weights = (upper-lower)/2*weights
    return nodes, weights

def prod_gauss(n1, n2):
    '''
    Product Gauss rule (Stroud C2: 3-1, 5-4, 7-4):
    
    The product form of a Gauss-Legendre quadrature rule.
    If n1 == n2 == n, this is a rule of order 2*n-1, using n**2 points.
    '''
    nodes1, weights1 = gauss_legendre(n1)
    nodes2, weights2 = gauss_legendre(n2)
    x_nodes = np.tile(nodes1, n2)
    y_nodes = np.repeat(nodes2, n1)
    weights = np.tile(weights1, n2) * np.repeat(weights2, n1)
    return (x_nodes, y_nodes), weights


print("1D Gauss quadrature")

expo = 8
f = lambda x: x**expo

print("{:15s}{:20s}{:20s}".format(" k","Integral","error"))
for k in range(1,14):
    ans = integrate.fixed_quad(f,0.0,1.0,n=k)
    error=abs(1/(expo+1)-ans[0])
    print("{:2d} {:20e} {:20e}".format(k,ans[0],error))
print("\n")

print("2D Gauss quadrature")
#g = lambda x,y: 0.075*np.log(0.3*x+0.5*y+4.2)# Burden chapter 4, example 2
g = lambda x,y: x**18*y**16

print("\n")
integ = integrate.dblquad(g,-1,1,-1,1)
print("Integral answer:",integ)
print("\n")

print("{:10s}{:20s}{:10s}{:10s}{:5s}".format(" k","Gauss-Legendre","order","npoints","error"))
for k in range(1,40,1):
    points = prod_gauss(k,k)
    npoints = len(points[0][0])
    order = 2*k-1
    integral = 0
    for i in range(0,npoints):
        weigth = points[1][i]
        integral += g(points[0][0][i],points[0][1][i]) * weigth 
    error = abs(integ[0]-integral)
    print("{:2d} {:20e} {:10d} {:10d} {:20e}".format(k,integral,order,npoints,error))
