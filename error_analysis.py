import numpy as np

from rk2_solver import rk2
from rk4_solver import rk4


def f(x, y):
    return y - x**2 + 1


def exact_solution(x):
    return (x + 1)**2 - 0.5 * np.exp(x)


def max_error(y_num, y_exact):
    return np.max(np.abs(y_num - y_exact))


x0 = 0.0
y0 = 0.5
x_final = 2.0

step_sizes = [0.4, 0.2, 0.1, 0.05]

print("Step size | RK2 max error | RK4 max error")
print("-" * 40)

for h in step_sizes:
    n = int((x_final - x0) / h)

    x_rk2, y_rk2 = rk2(f, x0, y0, h, n)
    x_rk4, y_rk4 = rk4(f, x0, y0, h, n)

    y_exact_rk2 = exact_solution(x_rk2)
    y_exact_rk4 = exact_solution(x_rk4)

    error_rk2 = max_error(y_rk2, y_exact_rk2)
    error_rk4 = max_error(y_rk4, y_exact_rk4)

    print(f"{h:<9} | {error_rk2:<13.6e} | {error_rk4:<13.6e}")
