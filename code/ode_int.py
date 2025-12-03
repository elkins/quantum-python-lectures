#!/usr/bin/env python3
"""ODE Integration Methods.

A system of ODEs (ordinary differential equations) of any order can always be
rewritten as a set of first-order equations.

This module provides various numerical methods for solving ODEs including:
- Explicit Euler (ee)
- Runge-Kutta 4th order (rk)
- Implicit Euler (ie)
- Adams-Bashforth 2-step (ab)

by Tommy Ogden <t.p.ogden@durham.ac.uk>
"""

import sys
from typing import Any, Callable, Optional, Union

import numpy as np


def ode_int(
    func: Callable[
        [Union[float, np.floating[Any]], np.ndarray, dict[str, Any]], np.ndarray
    ],
    y_0: np.ndarray,
    t: np.ndarray,
    method: str,
    args: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Approximate solution to a first-order ODE system with initial conditions.

    Args:
        func: The first-order ODE system to be approximated.
              Should have signature: func(t, y, args) -> dy/dt
        y_0: The initial condition array.
        t: A sequence of time points for which to solve for y.
        method: Integration method - 'ee' (Explicit Euler), 'rk' (Runge-Kutta),
                'ie' (Implicit Euler), or 'ab' (Adams-Bashforth)
        args: Extra arguments to pass to function (optional).

    Returns:
        The approximated solution of the system at each time in t,
        with the initial value y_0 in the first row.

    Raises:
        ValueError: If method is not one of the supported methods.
    """
    if args is None:
        args = {}

    if method == "ee":
        y = ode_int_ee(func, y_0, t, args)
    elif method == "rk":
        y = ode_int_rk(func, y_0, t, args)
    elif method == "ie":
        y = ode_int_ie(func, y_0, t, args)
    elif method == "ab":
        y = ode_int_ab(func, y_0, t, args)
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Supported methods are: 'ee', 'rk', 'ie', 'ab'"
        )

    return y


def ode_int_ee(
    func: Callable[
        [Union[float, np.floating[Any]], np.ndarray, dict[str, Any]], np.ndarray
    ],
    y_0: np.ndarray,
    t: np.ndarray,
    args: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Explicit Euler approximation to a first-order ODE system.

    A simple first-order method suitable for non-stiff problems.
    For each time step, approximates y(t+h) ≈ y(t) + h*f(t, y).

    Args:
        func: The first-order ODE system to be approximated.
        y_0: The initial condition array.
        t: A sequence of time points for which to solve for y.
        args: Extra arguments to pass to function (optional).

    Returns:
        The approximated solution of the system at each time in t,
        with the initial value y_0 in the first row.

    Example:
        >>> def exponential_decay(t, y, args):
        ...     return -args['k'] * y
        >>> y0 = np.array([1.0])
        >>> t = np.linspace(0, 2, 11)
        >>> result = ode_int_ee(exponential_decay, y0, t, {'k': 1.0})
        >>> result.shape
        (11, 1)
        >>> result[0, 0]  # Initial condition
        1.0
        >>> result[-1, 0] < 0.2  # Decayed after t=2
        True
    """
    if args is None:
        args = {}

    y = np.zeros([len(t), len(y_0)])  # Initialize the approximation array
    y[0] = y_0

    for i, t_i in enumerate(t[:-1]):  # Loop through the time steps
        h = t[i + 1] - t_i  # Size of the step
        y[i + 1] = y[i] + h * func(t_i, y[i], args)  # Euler step

    return y


def ode_int_rk(
    func: Callable[
        [Union[float, np.floating[Any]], np.ndarray, dict[str, Any]], np.ndarray
    ],
    y_0: np.ndarray,
    t: np.ndarray,
    args: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Classical Runge-Kutta (RK4) approximation to a first-order ODE system.

    A fourth-order method with excellent accuracy for smooth problems.
    Uses four function evaluations per step for higher accuracy than Euler.

    Args:
        func: The first-order ODE system to be approximated.
        y_0: The initial condition array.
        t: A sequence of time points for which to solve for y.
        args: Extra arguments to pass to function (optional).

    Returns:
        The approximated solution of the system at each time in t,
        with the initial value y_0 in the first row.

    Example:
        >>> def harmonic_oscillator(t, y, args):
        ...     # y[0] = position, y[1] = velocity
        ...     return np.array([y[1], -args['omega']**2 * y[0]])
        >>> y0 = np.array([1.0, 0.0])  # Start at x=1, v=0
        >>> t = np.linspace(0, 2*np.pi, 50)
        >>> result = ode_int_rk(harmonic_oscillator, y0, t, {'omega': 1.0})
        >>> result.shape
        (50, 2)
        >>> abs(result[-1, 0] - 1.0) < 0.01  # Should return to x≈1
        True
    """
    if args is None:
        args = {}

    # Initialize the approximation array
    y = np.zeros([len(t), len(y_0)])
    y[0] = y_0

    # Loop through the time steps, approximating this step from the prev step
    for i, t_i in enumerate(t[:-1]):
        h = t[i + 1] - t_i  # Size of the step

        k_1 = func(t_i, y[i], args)
        k_2 = func(t_i + h / 2.0, y[i] + h / 2.0 * k_1, args)
        k_3 = func(t_i + h / 2.0, y[i] + h / 2.0 * k_2, args)
        k_4 = func(t_i + h, y[i] + h * k_3, args)

        y[i + 1] = y[i] + h / 6.0 * (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4)  # RK4 step

    return y


def ode_int_ie(
    func: Callable[
        [Union[float, np.floating[Any]], np.ndarray, dict[str, Any]], np.ndarray
    ],
    y_0: np.ndarray,
    t: np.ndarray,
    args: Optional[dict[str, Any]] = None,
    max_iter: int = 10,
    tol: float = 1e-6,
) -> np.ndarray:
    """Implicit Euler approximation using Newton's method.

    Solves the nonlinear system y_{n+1} = y_n + h*f(t_{n+1}, y_{n+1}) using
    Newton iteration for stability on stiff problems.

    Args:
        func: The first-order ODE system to be approximated.
        y_0: The initial condition array.
        t: A sequence of time points for which to solve for y.
        args: Extra arguments to pass to function (optional).
        max_iter: Maximum number of Newton iterations per step (default: 10).
        tol: Convergence tolerance for Newton iteration (default: 1e-6).

    Returns:
        The approximated solution of the system at each time in t,
        with the initial value y_0 in the first row.

    Example:
        >>> def stiff_decay(t, y, args):
        ...     return -args['k'] * y  # Stiff for large k
        >>> y0 = np.array([1.0])
        >>> t = np.linspace(0, 1, 20)
        >>> # Explicit Euler would be unstable for k=100
        >>> result = ode_int_ie(stiff_decay, y0, t, {'k': 100.0})
        >>> result.shape
        (20, 1)
        >>> not np.any(np.isnan(result))  # Remains stable
        True
        >>> result[-1, 0] < 0.1  # Decays to near zero
        True
    """
    if args is None:
        args = {}

    # Initialize the approximation array
    y = np.zeros([len(t), len(y_0)])
    y[0] = y_0

    # Loop through the time steps
    for i, t_i in enumerate(t[:-1]):
        h = t[i + 1] - t_i  # Size of the step

        # Initial guess using explicit Euler
        y_guess = y[i] + h * func(t_i, y[i], args)

        # Newton iteration to solve: y_{n+1} - y_n - h*f(t_{n+1}, y_{n+1}) = 0
        for _ in range(max_iter):
            # Compute residual: G(y) = y - y_n - h*f(t_{n+1}, y)
            f_val = func(t[i + 1], y_guess, args)  # type: ignore[arg-type]
            residual = y_guess - y[i] - h * f_val

            # Check convergence
            if np.linalg.norm(residual) < tol:
                break

            # Approximate Jacobian using finite differences
            eps = np.sqrt(np.finfo(float).eps)
            jacobian = np.eye(len(y_0))
            for j in range(len(y_0)):
                y_pert = y_guess.copy()
                y_pert[j] += eps
                f_pert = func(t[i + 1], y_pert, args)  # type: ignore[arg-type]
                jacobian[:, j] = (f_pert - f_val) / eps

            # Compute Newton step: (I - h*J)*delta_y = -residual
            mat = np.eye(len(y_0)) - h * jacobian
            try:
                delta_y = np.linalg.solve(mat, -residual)
                y_guess = y_guess + delta_y
            except np.linalg.LinAlgError:
                # If singular, fall back to simple corrector step
                y_guess = y[i] + h * func(t[i + 1], y_guess, args)  # type: ignore[arg-type]
                break

        y[i + 1] = y_guess

    return y


def ode_int_ab(
    func: Callable[
        [Union[float, np.floating[Any]], np.ndarray, dict[str, Any]], np.ndarray
    ],
    y_0: np.ndarray,
    t: np.ndarray,
    args: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """Two-Step Adams-Bashforth approximation to a first-order ODE system.

    Args:
        func: The first-order ODE system to be approximated.
        y_0: The initial condition array.
        t: A sequence of time points for which to solve for y.
        args: Extra arguments to pass to function (optional).

    Returns:
        The approximated solution of the system at each time in t,
        with the initial value y_0 in the first row.
    """
    if args is None:
        args = {}

    # Initialize the approximation array
    y = np.zeros([len(t), len(y_0)])
    y[0] = y_0

    # Step 0: Euler
    h = t[1] - t[0]
    y[1] = y[0] + h * func(t[0], y[0], args)  # type: ignore[arg-type]  # Euler step

    # Step 1: Adams-Bashforth, Different Stepsizes
    h_1 = t[1] - t[0]
    h_2 = t[2] - t[1]

    y[2] = y[1] + 0.5 * h_2 / h_1 * (
        (2.0 * h_1 + h_2) * func(t[1], y[1], args)  # type: ignore[arg-type]
        - h_2 * func(t[0], y[0], args)  # type: ignore[arg-type]
    )

    # Steps 2 to N-1: Adams-Bashforth
    # Loop through the time steps
    for i, t_i in enumerate(t[2:-1], start=2):
        h = t[i + 1] - t_i  # Size of the step
        y[i + 1] = y[i] + (
            1.5 * h * func(t_i, y[i], args) - 0.5 * h * func(t[i - 1], y[i - 1], args)  # type: ignore[arg-type]
        )  # Adams-Bashforth

    return y


def main() -> int:
    """Main function for testing ODE integrators."""
    # Example usage could go here
    return 0


if __name__ == "__main__":
    sys.exit(main())
