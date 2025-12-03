#!/usr/bin/env python3
"""Smoke tests for modernized quantum-python-lectures code.

This script tests that the modernized Python code works correctly
with Python 3.9+ and has proper type hints.
"""

import sys
from pathlib import Path

import numpy as np

# Add code directory to path
code_dir = Path(__file__).parent / "code"
sys.path.insert(0, str(code_dir))


def test_ode_int():
    """Test all ODE integration methods."""
    import ode_int

    # Simple exponential decay: dy/dt = -y
    def test_func(t, y, args):
        return -y

    t = np.linspace(0, 2, 20)
    y0 = np.array([1.0])

    methods = ["ee", "rk", "ie", "ab"]
    results = {}

    for method in methods:
        result = ode_int.ode_int(test_func, y0, t, method)
        results[method] = result
        assert result.shape == (20, 1), f"{method} failed shape test"
        assert 0 < result[-1][0] < 1, f"{method} failed value test"

    print("✓ ode_int: All 4 methods tested successfully")
    return True


def test_lineshape_analysis():
    """Test lineshape analysis core functions."""
    try:
        import lineshape_analysis as la

        x = np.arange(-10, 10, 0.1)

        # Test gaussian
        g = la.gaussian(x, 0, 1)
        assert g.max() == 1.0, "Gaussian not normalized"
        assert len(g) == len(x), "Gaussian length mismatch"

        # Test lorentzian
        l = la.lorentzian(x, 0, 1)
        assert l.max() == 1.0, "Lorentzian not normalized"
        assert len(l) == len(x), "Lorentzian length mismatch"

        # Test voigt
        v = la.voigt(x, 0, 1, 0, 1)
        assert abs(v.max() - 1.0) < 0.01, f"Voigt not normalized (max={v.max()})"
        assert len(v) == len(x), "Voigt length mismatch"

        # Test generate_lineshapes
        yL, yG, yV = la.generate_lineshapes(x, 0, 1, 0, 1)
        assert len(yL) == len(yG) == len(yV) == len(x)

        print("✓ lineshape_analysis: All functions tested successfully")
        return True
    except ImportError as e:
        print(f"⚠ lineshape_analysis: Skipped (missing dependency: {e})")
        return False


def test_type_hints():
    """Check that type hints are present and Python 3.9 compatible."""
    from typing import get_type_hints

    import ode_int

    # Check that functions have type hints
    for func_name in [
        "ode_int",
        "ode_int_ee",
        "ode_int_rk",
        "ode_int_ie",
        "ode_int_ab",
    ]:
        func = getattr(ode_int, func_name)
        hints = get_type_hints(func)
        assert "return" in hints, f"{func_name} missing return type hint"
        assert len(hints) > 1, f"{func_name} missing parameter type hints"

    print("✓ Type hints: Present and compatible with Python 3.9")
    return True


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("SMOKE TESTS FOR MODERNIZED CODE")
    print("=" * 60)
    print(f"\nPython version: {sys.version}\n")

    tests = [
        test_ode_int,
        test_lineshape_analysis,
        test_type_hints,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
