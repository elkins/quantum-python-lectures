"""Performance benchmarks for ODE integration methods.

Compares accuracy and speed of different integration methods:
- Explicit Euler (ode_int_ee)
- Runge-Kutta 4 (ode_int_rk)
- Implicit Euler (ode_int_ie)
- Adams-Bashforth (ode_int_ab)

Also compares against SciPy's solve_ivp for reference.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add code directory to path
code_dir = Path(__file__).parent.parent / "code"
sys.path.insert(0, str(code_dir))

import ode_int

try:
    from scipy.integrate import solve_ivp

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# Test problems
def exponential_decay(t, y, args):
    """dy/dt = -k*y, solution: y = y0*exp(-k*t)"""
    k = args.get("k", 1.0)
    return -k * y


def harmonic_oscillator(t, y, args):
    """Simple harmonic oscillator: d2x/dt2 = -omega^2*x"""
    omega = args.get("omega", 1.0)
    return np.array([y[1], -(omega**2) * y[0]])


def stiff_problem(t, y, args):
    """Stiff decay: dy/dt = -k*y with large k"""
    k = args.get("k", 100.0)
    return -k * y


class TestExponentialDecayBenchmark:
    """Benchmark exponential decay problem."""

    def setup_method(self):
        """Set up test problem."""
        self.y0 = np.array([1.0])
        self.t = np.linspace(0, 2, 101)
        self.args = {"k": 1.0}

        # Analytical solution for comparison
        self.exact = np.exp(-self.args["k"] * self.t).reshape(-1, 1)

    def compute_error(self, result):
        """Compute maximum absolute error."""
        return np.max(np.abs(result - self.exact))

    def test_benchmark_explicit_euler(self, benchmark):
        """Benchmark explicit Euler method."""
        result = benchmark(
            ode_int.ode_int_ee, exponential_decay, self.y0, self.t, self.args
        )
        error = self.compute_error(result)
        print(f"\nExplicit Euler - Max Error: {error:.2e}")
        assert error < 0.1  # Reasonable for this step size

    def test_benchmark_runge_kutta(self, benchmark):
        """Benchmark Runge-Kutta 4 method."""
        result = benchmark(
            ode_int.ode_int_rk, exponential_decay, self.y0, self.t, self.args
        )
        error = self.compute_error(result)
        print(f"\nRunge-Kutta 4 - Max Error: {error:.2e}")
        assert error < 1e-6  # Should be very accurate

    def test_benchmark_implicit_euler(self, benchmark):
        """Benchmark implicit Euler method."""
        result = benchmark(
            ode_int.ode_int_ie, exponential_decay, self.y0, self.t, self.args
        )
        error = self.compute_error(result)
        print(f"\nImplicit Euler - Max Error: {error:.2e}")
        assert error < 0.2  # Less accurate but stable

    def test_benchmark_adams_bashforth(self, benchmark):
        """Benchmark Adams-Bashforth method."""
        result = benchmark(
            ode_int.ode_int_ab, exponential_decay, self.y0, self.t, self.args
        )
        error = self.compute_error(result)
        print(f"\nAdams-Bashforth - Max Error: {error:.2e}")
        assert error < 0.05  # Better than Euler

    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available")
    def test_benchmark_scipy_reference(self, benchmark):
        """Benchmark SciPy solve_ivp for comparison."""

        def func_scipy(t, y):
            return exponential_decay(t, y, self.args)

        def run_scipy():
            sol = solve_ivp(
                func_scipy,
                (self.t[0], self.t[-1]),
                self.y0,
                t_eval=self.t,
                method="RK45",
            )
            return sol.y.T

        result = benchmark(run_scipy)
        error = self.compute_error(result)
        print(f"\nSciPy RK45 - Max Error: {error:.2e}")


class TestHarmonicOscillatorBenchmark:
    """Benchmark harmonic oscillator problem."""

    def setup_method(self):
        """Set up test problem."""
        self.y0 = np.array([1.0, 0.0])  # x=1, v=0
        self.t = np.linspace(0, 2 * np.pi, 201)
        self.args = {"omega": 1.0}

        # Energy should be conserved
        self.initial_energy = 0.5 * (self.y0[0] ** 2 + self.y0[1] ** 2)

    def compute_energy_drift(self, result):
        """Compute maximum energy drift."""
        energies = 0.5 * (result[:, 0] ** 2 + result[:, 1] ** 2)
        drift = np.max(np.abs(energies - self.initial_energy))
        return drift

    def test_benchmark_rk4_oscillator(self, benchmark):
        """Benchmark RK4 on oscillator (should conserve energy well)."""
        result = benchmark(
            ode_int.ode_int_rk, harmonic_oscillator, self.y0, self.t, self.args
        )
        drift = self.compute_energy_drift(result)
        print(f"\nRK4 Oscillator - Energy Drift: {drift:.2e}")
        assert drift < 0.01


class TestStiffProblemBenchmark:
    """Benchmark stiff problem (tests stability)."""

    def setup_method(self):
        """Set up stiff test problem."""
        self.y0 = np.array([1.0])
        self.t = np.linspace(0, 1, 51)
        self.args = {"k": 100.0}

    def test_benchmark_implicit_euler_stiff(self, benchmark):
        """Benchmark implicit Euler on stiff problem."""
        result = benchmark(
            ode_int.ode_int_ie, stiff_problem, self.y0, self.t, self.args
        )
        # Should not blow up or produce NaN
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        print(f"\nImplicit Euler (stiff) - Final value: {result[-1, 0]:.2e}")

    def test_benchmark_rk4_stiff(self, benchmark):
        """Benchmark RK4 on stiff problem (for comparison)."""
        result = benchmark(
            ode_int.ode_int_rk, stiff_problem, self.y0, self.t, self.args
        )
        assert not np.any(np.isnan(result))
        print(f"\nRK4 (stiff) - Final value: {result[-1, 0]:.2e}")


class TestScalingBenchmark:
    """Test how methods scale with problem size."""

    def test_benchmark_rk4_scaling_2d(self, benchmark):
        """RK4 on 2D system."""
        y0 = np.array([1.0, 0.0])
        t = np.linspace(0, 10, 201)
        args = {"omega": 1.0}
        benchmark(ode_int.ode_int_rk, harmonic_oscillator, y0, t, args)

    def test_benchmark_rk4_scaling_10d(self, benchmark):
        """RK4 on 10D system."""

        def coupled_oscillators(t, y, args):
            """10D coupled oscillator system."""
            n = len(y) // 2
            dydt = np.zeros(2 * n)
            dydt[:n] = y[n:]  # velocities
            dydt[n:] = -y[:n]  # accelerations
            return dydt

        y0 = np.zeros(10)
        y0[0] = 1.0  # Initial displacement
        t = np.linspace(0, 10, 201)
        benchmark(ode_int.ode_int_rk, coupled_oscillators, y0, t, {})


if __name__ == "__main__":
    # Run benchmarks with detailed output
    pytest.main(
        [
            __file__,
            "-v",
            "--benchmark-only",
            "--benchmark-columns=mean,stddev,rounds",
            "--benchmark-sort=name",
        ]
    )
