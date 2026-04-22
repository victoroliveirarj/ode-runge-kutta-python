import numpy as np


def rk4(f, x0, y0, h, n):
    """
    Solve an ODE y' = f(x, y) using the classical fourth-order Runge-Kutta method.

    Parameters
    ----------
    f : function
        Right-hand side of the ODE, f(x, y).
    x0 : float
        Initial value of x.
    y0 : float
        Initial value of y.
    h : float
        Step size.
    n : int
        Number of steps.

    Returns
    -------
    x_values : numpy.ndarray
        Array of x values.
    y_values : numpy.ndarray
        Array of numerical y values.
    """
    x_values = np.zeros(n + 1)
    y_values = np.zeros(n + 1)

    x_values[0] = x0
    y_values[0] = y0

    for i in range(n):
        x = x_values[i]
        y = y_values[i]

        k1 = f(x, y)
        k2 = f(x + h / 2.0, y + h * k1 / 2.0)
        k3 = f(x + h / 2.0, y + h * k2 / 2.0)
        k4 = f(x + h, y + h * k3)

        y_values[i + 1] = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_values[i + 1] = x + h

    return x_values, y_values
