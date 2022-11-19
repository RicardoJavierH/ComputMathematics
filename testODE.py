import numpy as np
import matplotlib.pyplot as plt
import biblioteca as bib  

def f(y,t):
    dydt = y
    return dydt

y0 = 1
T = 4
t = np.linspace(0,T,10)
yE = bib.EulerODE(f,y0,t)
yRK = bib.RK4(f,y0,t)

tt = np.linspace(0,T,100)
yy = np.exp(tt)

plt.plot(t,yE,'*r')
plt.plot(t,yRK,'*g')
plt.plot(tt,yy,'b')
plt.show()


