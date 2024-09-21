import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

''' Datos del problema'''
# Omega = [a,b]x[a,b]
a = 0
b = 1
N = 4 # Número de puntos de discretización en cada eje

Pi = np.pi # Constante pi de numpy

def u(x,y): #solución exacta del problema de Poisson
    return np.sin(Pi*x)*np.sin(Pi*y)

def f(x,y): # Función fuente
    ans = 2*np.power(Pi,2)*np.sin(Pi*x)*np.sin(Pi*y)
    return ans

def g(x,y): # Condición de frontera
    return 0

'''Discretización del dominio omega'''
x = np.linspace(a,b,N)
y = np.linspace(a,b,N)
h = (b-a)/(N-1)

X, Y = np.meshgrid(x, y)
fig = plt.figure()
plt.plot(x[1],y[1],'ro',label='u no conocido');
plt.plot(X,Y,'ro');
plt.plot(np.ones(N),y,'bo',label='Condición de frontera');
plt.plot(np.zeros(N),y,'bo');
plt.plot(x,np.zeros(N),'bo');
plt.plot(x, np.ones(N),'bo');
plt.xlim((-0.1,1.1))
plt.ylim((-0.1,1.1))
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(r' Dominio discretizado $\Omega_h,$ N= %s'%(N),fontsize=16,y=1.08)
plt.show();

'''Ensamblaje del sistema de ecuaciones'''
N2=N*N
A=np.zeros((N2,N2))
## Diagonal            
for i in range (0,N2):           
    A[i,i]=4

# UPPER DIAGONAL        
for i in range (0,N2-1):          
    A[i,i+1]=-1   
# LOW DIAGONAL        
for i in range (1,N2-1):       
    A[i,i-1]=-1   

# LOWER IDENTITY MATRIX
for i in range (0,N):
    for j in range (1,N):           
        A[i+(N-1)*j,i+(N-1)*(j-1)]=-1 

# UPPER IDENTITY MATRIX
for i in range (0,N):
    for j in range (0,N-1):           
        A[i+(N-1)*j,i+(N-1)*(j+1)+1]=-1

print(A)     
r=np.zeros(N2)
 
# vector r      
for i in range (0,N-1):
    for j in range (0,N-1):           
        r[i+(N-1)*j]=100*h*h*(x[i+1]*x[i+1]+y[j+1]*y[j+1])     
# Boundary        
b_bottom_top=np.zeros(N2)
for i in range (0,N-1):
    b_bottom_top[i]=np.sin(2*np.pi*x[i+1]) #Bottom Boundary
    b_bottom_top[i+(N-1)*(N-2)]=np.sin(2*np.pi*x[i+1])# Top Boundary
      
b_left_right=np.zeros(N2)
for j in range (0,N-1):
    b_left_right[(N-1)*j]=2*np.sin(2*np.pi*y[j+1]) # Left Boundary
    b_left_right[N-2+(N-1)*j]=2*np.sin(2*np.pi*y[j+1])# Right Boundary
    
b=b_left_right+b_bottom_top

'''Resolución del sistema'''
w=np.zeros((N,N))

for i in range (0,N-1):
        w[i,0]=np.sin(2*np.pi*x[i]) #left Boundary
        w[i,N]=np.sin(2*np.pi*x[i]) #Right Boundary

for j in range (0,N-1):
        w[0,j]=2*np.sin(2*np.pi*y[j]) #Lower Boundary
        w[N,j]=2*np.sin(2*np.pi*y[j]) #Upper Boundary

Ainv=np.linalg.inv(A) 

C=np.dot(Ainv,r-b)
w[1:N,1:N]=C.reshape((N-1,N-1))

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d');
# Plot a basic wireframe.
ax.plot_wireframe(X, Y, w,color='r');
ax.set_xlabel('x');
ax.set_ylabel('y');
ax.set_zlabel('w');
plt.title(r'Numerical Approximation of the Poisson Equation',fontsize=24,y=1.08);
plt.show();

'''Gráfica de la solución aproximada'''

'''Gráfica de la solución exacta'''
Z = u(X,Y)

#Gráfica de densidad
plt.pcolormesh(X,Y,Z)
plt.colorbar()
plt.show()

# Gráfica de superficie
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.set_zlim(-1.01, 1.01) # Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.02f}') # A StrMethodFormatter is used automatically
fig.colorbar(surf, shrink=0.5, aspect=5) # Add a color bar which maps values to colors.
plt.show()