import sympy
import numpy as np
import matplotlib.pyplot as plt
#sympy.init_printing
#import scipy
from scipy import linalg as la
from scipy import optimize



''' sympy library'''

x = sympy.Symbol("x")
y, z = sympy.symbols("y,z")

expr1 = sympy.exp(sympy.cos(x))
der1 = expr1.diff(x)
der2 = sympy.Derivative(expr1,x)
der3 = sympy.Derivative(expr1,x,x)
der4 = sympy.Derivative(expr1,x,2)
print("der1: ", der1.doit())
print("der2: ", der2.doit())
print("der3: ", der3.doit())
print("der4: ", der3.doit())

expr2 = (x + 1)**3 * y ** 2 * (z - 1)
der5 = sympy.Derivative(expr2,x,y)
der6 = sympy.Derivative(expr2,y,x)

print("der5: ", der5.doit())
print("der6: ", der6.doit())

def f(x):
    return (x-1)**2

der=sympy.Derivative(f(x),x)
print(der.doit())

''' optimize module of scipy'''
def f(x):
    return [x[1] - x[0]**3 - 2 * x[0]**2 + 1, x[1] + x[0]**2 - 1]

sol = optimize.fsolve(f, [1, 1])
print(sol)

print("Evaluation in the numerical solution:",f(sol))

f_mat = sympy.Matrix([y - x**3 -2*x**2 + 1, y + x**2 - 1])
jac = f_mat.jacobian(sympy.Matrix([x, y]))
print(jac)


def f_jacobian(x):
    return [[-3*x[0]**2-4*x[0], 1], [2*x[0], 1]]

sol2 = optimize.fsolve(f, [1, 1], fprime=f_jacobian)
print(sol2)


def f(x):
   return [x[1] - x[0]**3 - 2 * x[0]**2 + 1, x[1] + x[0]**2 - 1]

x = np.linspace(-3, 2, 5000)
y1 = x**3 + 2 * x**2 -1
y2 = -x**2 + 1

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(x, y1, 'b', lw=1.5, label=r'$y = x^3 + 2x^2 - 1$')
ax.plot(x, y2, 'g', lw=1.5, label=r'$y = -x^2 + 1$')

x_guesses = [[-2, 2], [1, -1], [-2, -5]]
for x_guess in x_guesses:
    sol = optimize.fsolve(f, x_guess)
    ax.plot(sol[0], sol[1], 'r*', markersize=15)
    #ax.plot(x_guess[0], x_guess[1], 'ko')
    #ax.annotate( "", xy=(sol[0], sol[1]), xytext=(x_guess[0],
    #x_guess[1]), arrowprops=dict(arrowstyle="->", linewidth=2.5))

ax.legend(loc=0)
ax.set_xlabel(r'$x$', fontsize=18)
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y1, 'k', lw=1.5)
ax.plot(x, y2, 'k', lw=1.5)
sol1 = optimize.fsolve(f, [-2, 2])
sol2 = optimize.fsolve(f, [ 1, -1])
sol3 = optimize.fsolve(f, [-2, -5])
sols = [sol1, sol2, sol3]
colors = ['r', 'b', 'g']
for idx, s in enumerate(sols):
    ax.plot(s[0], s[1], colors[idx]+'*', markersize=15)

"Basin of attraction for the numerical method"
for m in np.linspace(-4, 3, 80):
    for n in np.linspace(-15, 15, 40):
        x_guess = [m, n]
        sol = optimize.fsolve(f, x_guess)
        idx = (abs(sols - sol)**2).sum(axis=1).argmin()
        ax.plot(x_guess[0], x_guess[1], colors[idx]+'.')

ax.set_xlabel(r'$x$', fontsize=18)
plt.show()

