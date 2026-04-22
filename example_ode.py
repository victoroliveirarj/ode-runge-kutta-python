import numpy as np
import matplotlib.pyplot as plt

from rk2_solver import rk2
from rk4_solver import rk4


def f(x, y):
    """
    Example ODE:
    y' = y - x**2 + 1
    """
    return y - x**2 + 1


def exact_solution(x):
    """
    Exact solution for:
    y' = y - x**2 + 1, with y(0) = 0.5
    """
    return (x + 1)**2 - 0.5 * np.exp(x)


# Initial condition
x0 = 0.0
y0 = 0.5

# Numerical setup
h = 0.2
x_final = 2.0
n = int((x_final - x0) / h)

# Compute numerical solutions
x_rk2, y_rk2 = rk2(f, x0, y0, h, n)
x_rk4, y_rk4 = rk4(f, x0, y0, h, n)

# Exact solution on a smooth grid
x_exact = np.linspace(x0, x_final, 300)
y_exact = exact_solution(x_exact)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x_exact, y_exact, label="Exact solution")
plt.plot(x_rk2, y_rk2, "o--", label="RK2")
plt.plot(x_rk4, y_rk4, "s--", label="RK4")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Runge-Kutta methods for an example ODE")
plt.legend()
plt.grid(True)
plt.show()
