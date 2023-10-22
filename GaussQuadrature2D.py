from scipy import integrate
import numpy as np

print("1D Gauss quadrature")
expo = 8
f = lambda x: x**expo

print("{:15s}{:20s}{:20s}".format(" k","Integral","error"))
for k in range(1,14):
    ans = integrate.fixed_quad(f,0.0,1.0,n=k)
    error=abs(1/(expo+1)-ans[0])
    print("{:2d} {:20e} {:20e}".format(k,ans[0],error))

print("2D Gauss quadrature")
g= lambda x,y: x**2*y**3

print("{:15s}{:20s}{:20s}".format(" k","Integral","error"))
for k in range(1,14):
    ans = integrate.dblquad(g,0,2,0,1)
    #error=abs(1/(expo+1)-ans[0])
    print("{:2d} {:20e} {:20e}".format(k,ans[0],ans[1]))
