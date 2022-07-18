from scipy.optimize import minimize
import sympy as sp

x2_symbol = sp.Symbol('x2')
u2_symbol = sp.Symbol('u2')
fm = 25 * u2_symbol - 20.0 * (sp.sin(x2_symbol)) + 38.7296387 * (sp.cos(x2_symbol)) - 38.7296387

def func(x):
    x2_float, u2_float = x
    return -fm.subs([(x2_symbol, x2_float), (u2_symbol, u2_float)])

def constraint1(x):
    x2_float, u2_float = x
    return -u2_float + 40 * sp.sin(x2_float) + 0.2

def constraint2(x):
    x2_float, u2_float = x
    return -u2_float - 40 * sp.sin(x2_float) + 0.2

b = [0, 1]
bounds = [b, b]
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
constraints = (con1, con2)
x0 = [0, 0]
solution = minimize(func, x0, method='SLSQP', bounds=bounds, constraints=constraints)
print(solution)
print(9)
