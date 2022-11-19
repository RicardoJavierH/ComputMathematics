import sympy

t, k, T0, Ta = sympy.symbols("t, k, T_0, T_a")
T = sympy.Function("T")
ode = T(t).diff(t) + k*(T(t) - Ta)
sympy.Eq(ode,0)
ode_sol = sympy.dsolve(ode)
print(ode_sol)

