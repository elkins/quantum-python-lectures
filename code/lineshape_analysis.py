"""Lineshape analysis functions for spectroscopy.

This module provides functions for generating and fitting common spectroscopic
lineshapes including Gaussian, Lorentzian, and Voigt profiles.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d as interp
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve as conv


def gaussian(x: np.ndarray, c: float, w: float) -> np.ndarray:
    """Analytic Gaussian function with center 'c' and width 'w'.

    Args:
            x: Input array of values
            c: Center position
            w: Width parameter (FWHM = 2*sqrt(2*log(2))*w)

    Returns:
            Normalized Gaussian array (peak = 1)

    Example:
        >>> x = np.linspace(-5, 5, 100)
        >>> y = gaussian(x, c=0, w=1.0)
        >>> y.max()  # Peak is normalized to 1
        1.0
        >>> abs(y[50] - 1.0) < 0.01  # Center at x=0
        True
    """
    G = np.exp(-((x - c) ** 2) / (2 * w**2))
    G /= G.max()
    return G


def lorentzian(x: np.ndarray, c: float, w: float) -> np.ndarray:
    """Analytic Lorentzian function with center 'c' and width 'w'.

    Args:
            x: Input array of values
            c: Center position
            w: Width parameter (FWHM = 2*w)

    Returns:
            Normalized Lorentzian array (peak = 1)

    Example:
        >>> x = np.linspace(-10, 10, 200)
        >>> y = lorentzian(x, c=0, w=1.0)
        >>> y.max()
        1.0
        >>> y[100]  # Center value at x=0
        1.0
        >>> y[0] > 0.01  # Heavy tails
        True
    """
    L = w**2 / ((x - c) ** 2 + w**2)
    L /= L.max()
    return L


def voigt(x: np.ndarray, c1: float, w1: float, c2: float, w2: float) -> np.ndarray:
    """Voigt function: convolution of Lorentzian and Gaussian.

    Convolution implemented with the FFT convolve function in scipy.

    Args:
            x: Input array of values
            c1: Lorentzian center position
            w1: Lorentzian width parameter
            c2: Gaussian center position
            w2: Gaussian width parameter

    Returns:
            Normalized Voigt array (peak = 1)
    """
    # Create larger array to avoid edge effects in convolution
    # Assumes monotonically increasing x-array
    dx = (x[-1] - x[0]) / len(x)
    xp_min = x[0] - len(x) / 3 * dx
    xp_max = x[-1] + len(x) / 3 * dx
    xp = np.linspace(xp_min, xp_max, 3 * len(x))

    L = lorentzian(xp, c1, w1)
    G = gaussian(xp, c2, w2)

    # Convolve
    V = conv(L, G, mode="same")

    # Normalize to unity height
    V /= V.max()

    # Create interpolation function to convert back to original array size
    fn_out = interp(xp, V)

    return fn_out(x)


def generate_lineshapes(
    x: np.ndarray, a: float, b: float, c: float, d: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate Lorentzian, Gaussian, and Voigt lineshapes.

    Args:
            x: Input array of values
            a: Lorentzian center
            b: Lorentzian width
            c: Gaussian center
            d: Gaussian width

    Returns:
            Tuple of (Lorentzian, Gaussian, Voigt) arrays
    """
    return lorentzian(x, a, b), gaussian(x, c, d), voigt(x, a, b, c, d)


def compare_lineshapes(wL: float, wG: float) -> None:
    """Create a plot comparing Voigt with Lorentzian and Gaussian.

    Args:
            wL: Lorentzian width
            wG: Gaussian width
    """
    # Generate some lineshape to analyze later
    x = np.arange(-20, 20, 0.01)
    yL, yG, yV = generate_lineshapes(x, 0, wL, 0, wG)
    y_noise = np.random.randn(len(x)) * 0.1
    yV += y_noise

    fig = plt.figure(2)
    plt.clf()

    ax = fig.add_subplot(111)

    ax.plot(x, yL / yL.max(), "b", lw=2, label="Lorentzian")
    ax.plot(x, yG / yG.max(), "r", lw=2, label="Gaussian")
    ax.plot(x, yV / yV.max(), "g--", lw=2, label="Voigt")

    # Add legend: loc='best' means find best position
    ax.legend(loc="best")

    ax.set_xlabel("Detuning (arb.)")
    ax.set_ylabel("Intensity (arb.)")


def fit_lineshape(x: np.ndarray, y: np.ndarray) -> None:
    """Fit data to Lorentzian, Gaussian, and Voigt functions and compare residuals.

    Args:
            x: Input array of x values
            y: Input array of y values (data to fit)
    """
    # Create figure panels using subplot2grid
    fig = plt.figure(1, figsize=(6, 7))
    fig.subplots_adjust(left=0.15, bottom=0.08, top=0.97)

    # Grid dimensions
    yy = 8
    xx = 8

    # Main panel
    ax = plt.subplot2grid((yy, xx), (0, 0), rowspan=yy - 3, colspan=xx - 1)

    # 3 residual panels
    ax_LRes = plt.subplot2grid(
        (yy, xx), (yy - 3, 0), rowspan=1, colspan=xx - 1, sharex=ax
    )
    ax_GRes = plt.subplot2grid(
        (yy, xx), (yy - 2, 0), rowspan=1, colspan=xx - 1, sharex=ax, sharey=ax_LRes
    )
    ax_VRes = plt.subplot2grid(
        (yy, xx), (yy - 1, 0), rowspan=1, colspan=xx - 1, sharex=ax, sharey=ax_LRes
    )

    # Residual histogram panels
    ax_LHist = plt.subplot2grid(
        (yy, xx), (yy - 3, xx - 1), rowspan=1, colspan=1, sharey=ax_LRes
    )
    ax_GHist = plt.subplot2grid(
        (yy, xx), (yy - 2, xx - 1), rowspan=1, colspan=1, sharey=ax_GRes
    )
    ax_VHist = plt.subplot2grid(
        (yy, xx), (yy - 1, xx - 1), rowspan=1, colspan=1, sharey=ax_VRes
    )

    # Turn off unwanted axes labels
    plt.setp(ax_LRes.get_xticklabels(), visible=False)
    plt.setp(ax_GRes.get_xticklabels(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax_LHist.get_yticklabels(), visible=False)
    plt.setp(ax_GHist.get_yticklabels(), visible=False)
    plt.setp(ax_VHist.get_yticklabels(), visible=False)
    plt.setp(ax_LHist.get_xticklabels(), visible=False)
    plt.setp(ax_GHist.get_xticklabels(), visible=False)
    plt.setp(ax_VHist.get_xticklabels(), visible=False)

    ax.set_ylabel("Intensity (arb.)")
    ax_VRes.set_xlabel("Detuning (arb.)")
    ax_LRes.set_ylabel("L.")
    ax_GRes.set_ylabel("G.")
    ax_VRes.set_ylabel("V.")

    # Plot initial data
    ax.plot(x, y, "k.", alpha=0.6)

    # FITTING

    # 1. Lorentzian
    pin = [0, 1]
    popt, perr = curve_fit(lorentzian, x, y, p0=pin)
    y_L = lorentzian(x, *popt)
    y_LRes = y - y_L

    # 2. Gaussian
    pin = [0, 1]
    popt, perr = curve_fit(gaussian, x, y, p0=pin)
    y_G = gaussian(x, *popt)
    y_GRes = y - y_G

    # 3. Voigt
    pin = [0, 1, 0, 1]
    popt, perr = curve_fit(voigt, x, y, p0=pin)
    y_V = voigt(x, *popt)
    y_VRes = y - y_V

    # PLOTTING

    # Add fits to main panel
    ax.plot(x, y_L, "b", lw=2, label="Lorentzian")
    ax.plot(x, y_G, "r", lw=2, label="Gaussian")
    ax.plot(x, y_V, "g--", lw=4, label="Voigt")

    ax.legend(loc="best")

    # Add residuals to sub-panels
    for axis in [ax_LRes, ax_GRes, ax_VRes, ax_LHist, ax_GHist, ax_VHist]:
        axis.axhline(0, color="k", linestyle="--")

    ax_LRes.plot(x, y_LRes, "b.")
    ax_GRes.plot(x, y_GRes, "r.")
    ax_VRes.plot(x, y_VRes, "g.")

    # Histograms
    histrange = (-0.15, 0.15)
    ax_LHist.hist(
        y_LRes, bins=25, orientation="horizontal", fc="b", alpha=0.6, range=histrange
    )
    ax_GHist.hist(
        y_GRes, bins=25, orientation="horizontal", fc="r", alpha=0.6, range=histrange
    )
    ax_VHist.hist(
        y_VRes, bins=25, orientation="horizontal", fc="g", alpha=0.6, range=histrange
    )

    plt.show()


def main(wL: float, wG: float) -> None:
    """Main function to generate and fit lineshape data.

    Args:
            wL: Lorentzian width
            wG: Gaussian width
    """
    # Generate data
    x = np.arange(-30, 30, 0.2)
    yL, yG, yV = generate_lineshapes(x, 0, wL, 0, wG)
    y_noise = np.random.randn(len(x)) * 0.03
    yV += y_noise

    fit_lineshape(x, yV)


if __name__ == "__main__":
    # Example usage
    main(wL=2.0, wG=1.5)
