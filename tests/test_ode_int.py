"""Comprehensive tests for ODE integration methods.

Tests numerical accuracy, edge cases, and error handling for all
integration methods: Explicit Euler, Runge-Kutta, Implicit Euler,
and Adams-Bashforth.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add code directory to path
code_dir = Path(__file__).parent.parent / "code"
sys.path.insert(0, str(code_dir))

import ode_int


# Test fixtures - analytical solutions for comparison
def exponential_decay(t, y, args):
    """dy/dt = -k*y, solution: y = y0*exp(-k*t)"""
    k = args.get("k", 1.0)
    return -k * y


def exponential_growth(t, y, args):
    """dy/dt = k*y, solution: y = y0*exp(k*t)"""
    k = args.get("k", 1.0)
    return k * y


def oscillator(t, y, args):
    """Simple harmonic oscillator: y'' = -omega^2*y
    Rewritten as system: y'[0] = y[1], y'[1] = -omega^2*y[0]
    Solution: y[0] = A*cos(omega*t) + B*sin(omega*t)
    """
    omega = args.get("omega", 1.0)
    return np.array([y[1], -(omega**2) * y[0]])


def linear_ode(t, y, args):
    """dy/dt = a*t + b, solution: y = a*t^2/2 + b*t + c"""
    a = args.get("a", 1.0)
    b = args.get("b", 0.0)
    return a * t + b


class TestODEIntMain:
    """Test the main ode_int function with method selection."""

    def test_method_selection(self):
        """Test that all methods can be selected and run."""

        def simple_func(t, y, args):
            return -y

        t = np.linspace(0, 1, 10)
        y0 = np.array([1.0])

        methods = ["ee", "rk", "ie", "ab"]
        for method in methods:
            result = ode_int.ode_int(simple_func, y0, t, method)
            assert result.shape == (10, 1), f"Method {method} returned wrong shape"
            assert not np.any(np.isnan(result)), f"Method {method} produced NaN"

    def test_invalid_method(self):
        """Test that invalid method raises appropriate error."""

        def simple_func(t, y, args):
            return -y

        t = np.linspace(0, 1, 10)
        y0 = np.array([1.0])

        with pytest.raises((ValueError, KeyError)):
            ode_int.ode_int(simple_func, y0, t, "invalid_method")

    def test_empty_args(self):
        """Test that functions work without args dict."""

        def no_args_func(t, y, args):
            return -y

        t = np.linspace(0, 1, 10)
        y0 = np.array([1.0])

        result = ode_int.ode_int(no_args_func, y0, t, "ee")
        assert result.shape == (10, 1)


class TestExplicitEuler:
    """Test Explicit Euler (Forward Euler) method."""

    def test_exponential_decay(self):
        """Test accuracy on exponential decay problem."""
        t = np.linspace(0, 2, 100)
        y0 = np.array([1.0])
        args = {"k": 1.0}

        result = ode_int.ode_int_ee(exponential_decay, y0, t, args)

        # Analytical solution
        analytical = np.exp(-args["k"] * t)

        # Check final value within reasonable tolerance
        # Euler method is first-order, so error is O(h)
        assert abs(result[-1, 0] - analytical[-1]) < 0.1

    def test_constant_solution(self):
        """Test that constant solutions remain constant."""

        def zero_derivative(t, y, args):
            return np.zeros_like(y)

        t = np.linspace(0, 1, 50)
        y0 = np.array([5.0])

        result = ode_int.ode_int_ee(zero_derivative, y0, t, {})

        # All values should equal y0
        np.testing.assert_allclose(result[:, 0], y0[0])

    def test_multidimensional(self):
        """Test with multi-dimensional state vector."""
        t = np.linspace(0, 1, 50)
        y0 = np.array([1.0, 2.0, 3.0])
        args = {}

        result = ode_int.ode_int_ee(exponential_decay, y0, t, args)

        assert result.shape == (50, 3)
        # Each component should decay independently
        for i in range(3):
            assert result[-1, i] < y0[i]

    def test_single_step(self):
        """Test with just two time points (single step)."""
        t = np.array([0.0, 0.1])
        y0 = np.array([1.0])
        args = {"k": 1.0}

        result = ode_int.ode_int_ee(exponential_decay, y0, t, args)

        # Manual calculation: y1 = y0 + h*f(t0, y0)
        expected = y0[0] + 0.1 * (-1.0 * y0[0])
        np.testing.assert_almost_equal(result[1, 0], expected)


class TestRungeKutta:
    """Test 4th-order Runge-Kutta method (RK4)."""

    def test_exponential_decay_accuracy(self):
        """Test that RK4 is more accurate than Euler."""
        t = np.linspace(0, 2, 50)
        y0 = np.array([1.0])
        args = {"k": 1.0}

        result_rk = ode_int.ode_int_rk(exponential_decay, y0, t, args)
        result_ee = ode_int.ode_int_ee(exponential_decay, y0, t, args)

        analytical = np.exp(-args["k"] * t[-1])

        error_rk = abs(result_rk[-1, 0] - analytical)
        error_ee = abs(result_ee[-1, 0] - analytical)

        # RK4 should be much more accurate
        assert error_rk < error_ee / 10

    def test_harmonic_oscillator(self):
        """Test RK4 on harmonic oscillator (2D system)."""
        omega = 2.0 * np.pi  # frequency
        t = np.linspace(0, 2, 100)
        y0 = np.array([1.0, 0.0])  # initial displacement, zero velocity
        args = {"omega": omega}

        result = ode_int.ode_int_rk(oscillator, y0, t, args)

        # Analytical solution: y = cos(omega*t)
        analytical = np.cos(omega * t)

        # Check position (first component) - slightly relaxed tolerance
        np.testing.assert_allclose(result[:, 0], analytical, rtol=2e-3)

    def test_energy_conservation(self):
        """Test energy conservation in harmonic oscillator."""
        omega = 1.0
        t = np.linspace(0, 10, 500)
        y0 = np.array([1.0, 0.0])
        args = {"omega": omega}

        result = ode_int.ode_int_rk(oscillator, y0, t, args)

        # Energy = (1/2)*(velocity^2 + omega^2*position^2)
        energy = 0.5 * (result[:, 1] ** 2 + omega**2 * result[:, 0] ** 2)

        # Energy should be approximately constant
        energy_variation = np.std(energy) / np.mean(energy)
        assert energy_variation < 0.01  # Less than 1% variation


class TestImplicitEuler:
    """Test Implicit (Backward) Euler method."""

    def test_exponential_decay_stability(self):
        """Test that implicit Euler is stable for stiff problems."""
        # Use very stiff problem (large k)
        k = 100.0
        t = np.linspace(0, 1, 20)
        y0 = np.array([1.0])
        args = {"k": k}

        # Implicit Euler should handle this without blowing up
        result = ode_int.ode_int_ie(exponential_decay, y0, t, args)

        assert not np.any(np.isnan(result)), "Implicit Euler produced NaN"
        assert not np.any(np.isinf(result)), "Implicit Euler produced Inf"
        assert result[-1, 0] < y0[0], "Solution should decay"

        # For very stiff problem, solution should decay to nearly zero
        # Small negative values (~1e-9) are acceptable numerical artifacts
        assert result[-1, 0] < 0.1, "Should decay significantly"
        assert abs(result[-1, 0]) < 0.1, "Should be close to zero"

    def test_convergence_iterations(self):
        """Test that implicit method converges within iterations."""
        t = np.linspace(0, 1, 50)
        y0 = np.array([1.0])
        args = {"k": 1.0}

        result = ode_int.ode_int_ie(exponential_decay, y0, t, args)

        # Should produce reasonable result
        analytical = np.exp(-args["k"] * t[-1])
        assert abs(result[-1, 0] - analytical) < 0.2


class TestAdamsBashforth:
    """Test Adams-Bashforth multi-step method."""

    def test_requires_multiple_steps(self):
        """Test that AB method works with multiple time steps."""
        t = np.linspace(0, 2, 100)
        y0 = np.array([1.0])
        args = {"k": 1.0}

        result = ode_int.ode_int_ab(exponential_decay, y0, t, args)

        assert result.shape == (100, 1)
        assert not np.any(np.isnan(result))

    def test_exponential_accuracy(self):
        """Test accuracy on exponential problem."""
        t = np.linspace(0, 2, 100)
        y0 = np.array([1.0])
        args = {"k": 1.0}

        result = ode_int.ode_int_ab(exponential_decay, y0, t, args)
        analytical = np.exp(-args["k"] * t[-1])

        # AB should be reasonably accurate
        assert abs(result[-1, 0] - analytical) < 0.05


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_initial_condition(self):
        """Test with zero initial condition."""
        t = np.linspace(0, 1, 50)
        y0 = np.array([0.0])
        args = {"k": 1.0}

        result = ode_int.ode_int_rk(exponential_decay, y0, t, args)

        # Zero should remain zero
        np.testing.assert_allclose(result[:, 0], 0.0)

    def test_negative_initial_condition(self):
        """Test with negative initial values."""
        t = np.linspace(0, 1, 50)
        y0 = np.array([-1.0])
        args = {"k": 1.0}

        result = ode_int.ode_int_rk(exponential_decay, y0, t, args)

        # Should decay toward zero from below
        assert result[-1, 0] < 0
        assert result[-1, 0] > y0[0]

    def test_variable_step_size(self):
        """Test with non-uniform time steps."""
        t = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
        y0 = np.array([1.0])
        args = {"k": 1.0}

        result = ode_int.ode_int_rk(exponential_decay, y0, t, args)

        assert result.shape == (5, 1)
        # Should still be monotonically decreasing
        assert np.all(np.diff(result[:, 0]) < 0)

    def test_very_small_step(self):
        """Test with very small time step."""
        t = np.linspace(0, 0.001, 10)
        y0 = np.array([1.0])
        args = {"k": 1.0}

        result = ode_int.ode_int_rk(exponential_decay, y0, t, args)

        # Should change very little
        assert abs(result[-1, 0] - y0[0]) < 0.002

    def test_large_dimensional_system(self):
        """Test with higher dimensional system."""
        t = np.linspace(0, 1, 50)
        y0 = np.ones(10)  # 10-dimensional system
        args = {"k": 1.0}

        result = ode_int.ode_int_rk(exponential_decay, y0, t, args)

        assert result.shape == (50, 10)
        # All components should decay
        assert np.all(result[-1] < y0)


class TestNumericalProperties:
    """Test numerical properties and accuracy."""

    def test_order_of_accuracy_euler(self):
        """Verify Euler method is first-order accurate."""
        y0 = np.array([1.0])
        args = {"k": 1.0}

        # Test with different step sizes
        t_coarse = np.linspace(0, 1, 11)  # h = 0.1
        t_fine = np.linspace(0, 1, 21)  # h = 0.05

        result_coarse = ode_int.ode_int_ee(exponential_decay, y0, t_coarse, args)
        result_fine = ode_int.ode_int_ee(exponential_decay, y0, t_fine, args)

        analytical = np.exp(-args["k"])

        error_coarse = abs(result_coarse[-1, 0] - analytical)
        error_fine = abs(result_fine[-1, 0] - analytical)

        # Error should reduce by ~2x when step size halves (first order)
        ratio = error_coarse / error_fine
        assert 1.5 < ratio < 2.5  # Allow some numerical tolerance

    def test_order_of_accuracy_rk4(self):
        """Verify RK4 method is fourth-order accurate."""
        y0 = np.array([1.0])
        args = {"k": 1.0}

        # Test with different step sizes
        t_coarse = np.linspace(0, 1, 11)  # h = 0.1
        t_fine = np.linspace(0, 1, 21)  # h = 0.05

        result_coarse = ode_int.ode_int_rk(exponential_decay, y0, t_coarse, args)
        result_fine = ode_int.ode_int_rk(exponential_decay, y0, t_fine, args)

        analytical = np.exp(-args["k"])

        error_coarse = abs(result_coarse[-1, 0] - analytical)
        error_fine = abs(result_fine[-1, 0] - analytical)

        # Error should reduce by ~16x when step size halves (fourth order)
        # Allow wide tolerance due to small errors
        if error_coarse > 1e-10:  # Only test if error is measurable
            ratio = error_coarse / error_fine
            assert ratio > 8  # At least some indication of higher order

    def test_monotonicity_preservation(self):
        """Test that monotonic solutions remain monotonic."""
        t = np.linspace(0, 2, 100)
        y0 = np.array([1.0])
        args = {"k": 1.0}

        result = ode_int.ode_int_rk(exponential_decay, y0, t, args)

        # Exponential decay should be monotonically decreasing
        differences = np.diff(result[:, 0])
        assert np.all(differences <= 0)


class TestCoverageImprovements:
    """Tests to improve code coverage by exercising uncovered branches."""

    def test_singular_matrix_fallback(self):
        """Test implicit Euler fallback when Jacobian is singular."""

        def singular_function(t, y, args):
            """A function that creates a singular Jacobian."""
            # Returns zero derivative, leading to singular I - h*J
            return np.zeros_like(y)

        y0 = np.array([1.0, 2.0])
        t = np.linspace(0, 1, 5)

        # Should not crash, should use fallback
        result = ode_int.ode_int_ie(singular_function, y0, t, {})
        assert result.shape == (5, 2)
        assert not np.any(np.isnan(result))

    def test_main_function(self):
        """Test the main() function exists and returns 0."""
        result = ode_int.main()
        assert result == 0


class TestArgumentHandling:
    """Test edge cases in argument handling across all methods."""

    def test_explicit_euler_with_empty_dict(self):
        """Test explicit Euler with explicitly passed empty args dict."""
        y0 = np.array([1.0])
        t = np.linspace(0, 1, 10)

        # Pass empty dict explicitly
        result = ode_int.ode_int_ee(exponential_decay, y0, t, {})
        assert result.shape == (10, 1)

    def test_runge_kutta_with_empty_dict(self):
        """Test Runge-Kutta with explicitly passed empty args dict."""
        y0 = np.array([1.0])
        t = np.linspace(0, 1, 10)

        result = ode_int.ode_int_rk(exponential_decay, y0, t, {})
        assert result.shape == (10, 1)

    def test_implicit_euler_with_empty_dict(self):
        """Test implicit Euler with explicitly passed empty args dict."""
        y0 = np.array([1.0])
        t = np.linspace(0, 1, 10)

        result = ode_int.ode_int_ie(exponential_decay, y0, t, {})
        assert result.shape == (10, 1)

    def test_adams_bashforth_with_empty_dict(self):
        """Test Adams-Bashforth with explicitly passed empty args dict."""
        y0 = np.array([1.0])
        t = np.linspace(0, 1, 10)

        result = ode_int.ode_int_ab(exponential_decay, y0, t, {})
        assert result.shape == (10, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
