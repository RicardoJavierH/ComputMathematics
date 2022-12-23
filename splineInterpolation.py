import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def phi(j,x,nodVec):
    h=nodVec[1]-nodVec[0]
    
    def tramo1(x):
        return 0
    def tramo2(x):
        return np.power((x-nodVec[j-2]),3)/(6*np.power(h,3))
    def tramo3(x):
        resp4a = 2/3-np.power((x-nodVec[j]),2)/np.power(h,2)
        resp4b = -np.power((x-nodVec[j]),3)/(2*np.power(h,3))
        return resp4a + resp4b
    def tramo4(x):
        resp3a = 1/6+(nodVec[j+1]-x)/(2*h)+np.power((nodVec[j+1]-x),2)/(2*np.power(h,2))
        resp3b =-np.power((nodVec[j+1]-x),3)/(2*np.power(h,3))
        return resp3a+resp3b
    
    def tramo5(x):
        return np.power((nodVec[j+2]-x),3)/(6*np.power(h,3))
    def tramo6(x):
        return 0
    return np.piecewise(x,[(x<nodVec[j-2]),
    (nodVec[j-2]<=x) & (x<=nodVec[j-1]),(nodVec[j-1]<=x) & (x<=nodVec[j]),
    (nodVec[j]<=x) & (x<=nodVec[j+1]),(nodVec[j+1]<=x) & (x<=nodVec[j+2]),(x>nodVec[j+2])],[lambda x:tramo1(x),lambda x: tramo2(x),
    lambda x:tramo3(x),lambda x:tramo4(x),lambda x:tramo5(x),lambda x:tramo6(x)])


a = 0
b = 2
nNodos = 8
nodVec = np.linspace(a,b,nNodos)
h=nodVec[1]-nodVec[0]

xx = np.linspace(a-h,b+h,1000)
j=6
yy = phi(j,xx,nodVec)
#yexact = np.arctan(xx) 
plt.plot(xx,yy)
'''
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(xx,yexact,'g',lw=2,label='Solución exacta') 
ax.plot(xx,yy,'b',lw=2,label='Polinomio interpolante')
ax.plot(x,y,'ro',alpha=0.6,label='Datos')
ax.legend(loc=2)
ax.set_xlabel(r"$x$", fontsize=10)
ax.set_ylabel(r"$y$", fontsize=10)
'''
plt.grid()
plt.show()