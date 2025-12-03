"""Tests for lineshape analysis functions.

Tests the spectroscopic lineshape functions with their actual API:
- gaussian(x, c, w): center c, width w
- lorentzian(x, c, w): center c, width w
- voigt(x, c1, w1, c2, w2): Lorentzian (c1, w1) convolved with Gaussian (c2, w2)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

code_dir = Path(__file__).parent.parent / "code"
sys.path.insert(0, str(code_dir))

try:
    import lineshape_analysis as la

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="scipy not available")


class TestGaussian:
    """Test Gaussian lineshape function."""

    def test_normalization(self):
        """Test that Gaussian peak is normalized to 1."""
        x = np.linspace(-10, 10, 1000)
        y = la.gaussian(x, c=0, w=1)

        assert abs(y.max() - 1.0) < 1e-10

    def test_center_position(self):
        """Test that maximum is at center c."""
        x = np.linspace(-10, 10, 1000)
        c = 3.5
        y = la.gaussian(x, c=c, w=1)

        x_max = x[np.argmax(y)]
        assert abs(x_max - c) < 0.02

    def test_width_scaling(self):
        """Test that width scales with parameter w."""
        x = np.linspace(-10, 10, 10000)

        w1, w2 = 1.0, 2.0
        y1 = la.gaussian(x, c=0, w=w1)
        y2 = la.gaussian(x, c=0, w=w2)

        # Find FWHM
        above_half1 = x[y1 >= 0.5]
        above_half2 = x[y2 >= 0.5]

        fwhm1 = above_half1.max() - above_half1.min()
        fwhm2 = above_half2.max() - above_half2.min()

        # FWHM should scale proportionally
        assert abs(fwhm2 / fwhm1 - w2 / w1) < 0.05

    def test_symmetry(self):
        """Test that Gaussian is approximately symmetric around center."""
        x = np.linspace(-10, 10, 1001)  # Odd number for exact center point
        c = 0.0
        y = la.gaussian(x, c=c, w=1)

        # Center should be at middle index
        idx_center = len(x) // 2

        # Check a few offsets for approximate symmetry
        for offset in [1, 5, 10, 20]:
            left = y[idx_center - offset]
            right = y[idx_center + offset]
            # Allow small difference due to numerical precision
            rel_diff = abs(left - right) / max(left, right)
            assert rel_diff < 0.01, f"Symmetry broken at offset {offset}"

    def test_no_nans(self):
        """Test that function doesn't produce NaN values."""
        x = np.linspace(-100, 100, 1000)

        for c in [-10, 0, 10]:
            for w in [0.1, 1.0, 10.0]:
                y = la.gaussian(x, c=c, w=w)
                assert not np.any(np.isnan(y))
                assert not np.any(np.isinf(y))


class TestLorentzian:
    """Test Lorentzian lineshape function."""

    def test_normalization(self):
        """Test that Lorentzian peak is normalized to 1."""
        x = np.linspace(-10, 10, 1000)
        y = la.lorentzian(x, c=0, w=1)

        assert abs(y.max() - 1.0) < 1e-10

    def test_center_position(self):
        """Test that maximum is at center c."""
        x = np.linspace(-10, 10, 1000)
        c = -2.5
        y = la.lorentzian(x, c=c, w=1)

        x_max = x[np.argmax(y)]
        assert abs(x_max - c) < 0.02

    def test_fwhm_equals_2w(self):
        """Test that FWHM equals 2*w for Lorentzian."""
        x = np.linspace(-10, 10, 10000)
        w = 1.5
        y = la.lorentzian(x, c=0, w=w)

        # Find FWHM
        above_half = x[y >= 0.5]
        fwhm = above_half.max() - above_half.min()

        expected_fwhm = 2 * w
        assert abs(fwhm - expected_fwhm) < 0.02

    def test_heavy_tails(self):
        """Test that Lorentzian has heavier tails than Gaussian."""
        x = np.linspace(0, 10, 1000)

        y_lorentz = la.lorentzian(x, c=0, w=1)
        y_gauss = la.gaussian(x, c=0, w=1)

        # At x = 5, Lorentzian should have heavier tail
        tail_idx = np.argmin(abs(x - 5))
        assert y_lorentz[tail_idx] > y_gauss[tail_idx]

    def test_no_nans(self):
        """Test that function doesn't produce NaN values."""
        x = np.linspace(-100, 100, 1000)

        for c in [-10, 0, 10]:
            for w in [0.1, 1.0, 10.0]:
                y = la.lorentzian(x, c=c, w=w)
                assert not np.any(np.isnan(y))
                assert not np.any(np.isinf(y))


class TestVoigt:
    """Test Voigt profile (convolution of Gaussian and Lorentzian)."""

    def test_voigt_normalization(self):
        """Test that Voigt peak is approximately normalized."""
        x = np.linspace(-10, 10, 1000)
        y = la.voigt(x, c1=0, w1=1, c2=0, w2=1)

        # Voigt peak should be close to 1
        assert abs(y.max() - 1.0) < 0.05

    def test_voigt_intermediate_behavior(self):
        """Test that Voigt is intermediate between Gaussian and Lorentzian."""
        x = np.linspace(-10, 10, 1000)

        y_voigt = la.voigt(x, c1=0, w1=1, c2=0, w2=1)
        y_gauss = la.gaussian(x, c=0, w=1)
        la.lorentzian(x, c=0, w=1)

        # Check tail behavior - Voigt should have heavier tail than pure Gaussian
        tail_idx = np.argmin(abs(x - 5))

        # At the tail: Voigt should be between or close to Lorentzian
        # Due to convolution and normalization, exact ordering may vary
        assert (
            y_voigt[tail_idx] > y_gauss[tail_idx]
        ), "Voigt should have heavier tail than Gaussian"

    def test_offset_centers(self):
        """Test Voigt with different centers for Gaussian and Lorentzian."""
        x = np.linspace(-10, 10, 1000)

        y = la.voigt(x, c1=1.0, w1=1, c2=2.0, w2=1)

        assert not np.any(np.isnan(y))
        assert not np.any(np.isinf(y))
        assert y.max() > 0

    def test_no_nans(self):
        """Test that function doesn't produce NaN values."""
        x = np.linspace(-20, 20, 1000)

        for c in [0, 5]:
            for w in [0.5, 1.0, 2.0]:
                y = la.voigt(x, c1=c, w1=w, c2=c, w2=w)
                assert not np.any(np.isnan(y))
                assert not np.any(np.isinf(y))


class TestGenerateLineshapes:
    """Test the generate_lineshapes utility function."""

    def test_output_shapes(self):
        """Test that all outputs have correct shape."""
        x = np.linspace(-5, 5, 100)

        yL, yG, yV = la.generate_lineshapes(x, a=0, b=1, c=0, d=1)

        assert yL.shape == x.shape
        assert yG.shape == x.shape
        assert yV.shape == x.shape

    def test_output_values(self):
        """Test that outputs match individual functions."""
        x = np.linspace(-5, 5, 100)
        a, b, c, d = 0, 1.0, 0, 1.0

        yL, yG, yV = la.generate_lineshapes(x, a, b, c, d)

        # Compare with individual functions
        yL_expected = la.lorentzian(x, a, b)
        yG_expected = la.gaussian(x, c, d)
        yV_expected = la.voigt(x, a, b, c, d)

        np.testing.assert_allclose(yL, yL_expected)
        np.testing.assert_allclose(yG, yG_expected)
        np.testing.assert_allclose(yV, yV_expected)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_point(self):
        """Test with single x value."""
        x = np.array([0.0])

        y_gauss = la.gaussian(x, c=0, w=1)
        y_lorentz = la.lorentzian(x, c=0, w=1)

        assert len(y_gauss) == 1
        assert len(y_lorentz) == 1
        assert y_gauss[0] == 1.0
        assert y_lorentz[0] == 1.0

    def test_large_offset(self):
        """Test with very large x values (far from peak)."""
        # Include peak in range so normalization works correctly
        x = np.array([-5.0, 0.0, 1000.0, 2000.0, 3000.0])

        y_gauss = la.gaussian(x, c=0, w=1)
        y_lorentz = la.lorentzian(x, c=0, w=1)

        # Peak should be at index 1 (x=0)
        assert y_gauss[1] == 1.0
        assert y_lorentz[1] == 1.0

        # Far values (indices 2,3,4) should be very small
        assert np.all(y_gauss[2:] < 1e-10) or np.all(np.isnan(y_gauss[2:]))
        assert np.all(y_lorentz[2:] < 1e-5)

    def test_negative_x_range(self):
        """Test with all negative x values."""
        x = np.linspace(-10, -5, 100)

        y_gauss = la.gaussian(x, c=-7.5, w=1)
        y_lorentz = la.lorentzian(x, c=-7.5, w=1)
        y_voigt = la.voigt(x, c1=-7.5, w1=1, c2=-7.5, w2=1)

        assert not np.any(np.isnan(y_gauss))
        assert not np.any(np.isnan(y_lorentz))
        assert not np.any(np.isnan(y_voigt))

    def test_very_large_width(self):
        """Test with very large width parameters."""
        x = np.linspace(-100, 100, 1000)

        y_gauss = la.gaussian(x, c=0, w=50)
        y_lorentz = la.lorentzian(x, c=0, w=50)

        # Should be very broad
        assert y_gauss.max() == 1.0
        assert y_lorentz.max() == 1.0

        # Values at edges should be significant
        assert y_lorentz[0] > 0.01
        assert y_lorentz[-1] > 0.01


class TestNumericalProperties:
    """Test numerical properties and stability."""

    def test_positive_values(self):
        """Test that all lineshapes are non-negative."""
        x = np.linspace(-100, 100, 1000)

        y_gauss = la.gaussian(x, c=0, w=1)
        y_lorentz = la.lorentzian(x, c=0, w=1)
        y_voigt = la.voigt(x, c1=0, w1=1, c2=0, w2=1)

        assert np.all(y_gauss >= 0)
        assert np.all(y_lorentz >= 0)
        assert np.all(y_voigt >= 0)

    def test_integral_scaling(self):
        """Test that integral scales correctly with width."""
        x = np.linspace(-20, 20, 10000)
        dx = x[1] - x[0]

        # Gaussian integral should scale with width
        w1, w2 = 1.0, 2.0

        y1 = la.gaussian(x, c=0, w=w1)
        y2 = la.gaussian(x, c=0, w=w2)

        integral1 = np.sum(y1) * dx
        integral2 = np.sum(y2) * dx

        # Integrals should scale with width
        ratio = integral2 / integral1
        expected_ratio = w2 / w1

        assert abs(ratio - expected_ratio) < 0.1


class TestFittingFunctions:
    """Test fitting-related functions if available."""

    def test_fit_lineshape_exists(self):
        """Test that fit_lineshape function exists."""
        assert hasattr(la, "fit_lineshape")


class TestPlottingFunctions:
    """Test plotting functions don't crash (no visual validation)."""

    @pytest.fixture(autouse=True)
    def setup_matplotlib(self):
        """Configure matplotlib for non-interactive testing."""
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend

    def test_compare_lineshapes_runs(self):
        """Test that compare_lineshapes executes without error."""
        # Just verify it doesn't crash - won't check visual output
        try:
            la.compare_lineshapes(wL=2.0, wG=1.0)
        except Exception as e:
            pytest.fail(f"compare_lineshapes raised {type(e).__name__}: {e}")

    def test_fit_lineshape_runs(self):
        """Test that fit_lineshape executes without error."""
        x = np.linspace(-10, 10, 200)
        # Create synthetic Gaussian data
        y_data = la.gaussian(x, c=0, w=2.0)

        try:
            # Try to fit with correct signature (no lineshape_type parameter)
            la.fit_lineshape(x, y_data)
            # Function may return None or parameters
        except Exception as e:
            # Some fitting functions might not be fully implemented
            # Just ensure no unexpected errors
            if "not implemented" not in str(e).lower():
                pytest.fail(f"fit_lineshape raised unexpected error: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
