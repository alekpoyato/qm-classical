"""Create sampled 1D wavefunctions on a uniform position grid.

A state is a complex NumPy array psi; its grid x and hbar are explicit.
The interval is periodic for future FFT use; keep packets far from its edges.
"""

import numpy as np


def position_grid(x_min=-10.0, x_max=10.0, points=1024):
    """Return equally spaced positions, excluding the right endpoint."""
    if points < 2 or x_max <= x_min:
        raise ValueError("Use at least two points and x_max > x_min.")
    return np.linspace(x_min, x_max, points, endpoint=False)


def grid_spacing(x):
    """Check that x is an increasing uniform grid and return dx."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("x must contain at least two finite positions.")
    steps = np.diff(x)
    if steps[0] <= 0 or not np.allclose(steps, steps[0]):
        raise ValueError("x must be increasing and uniformly spaced.")
    return float(steps[0])


def probability_density(psi):
    """Return |psi|^2: a density, not the probability per grid point."""
    return np.abs(psi) ** 2


def normalize(psi, dx):
    """Return a new array with dx * sum(|psi|^2) equal to one."""
    psi = np.asarray(psi, dtype=complex)
    norm = dx * np.sum(probability_density(psi))
    if dx <= 0 or not np.isfinite(norm) or norm <= 0:
        raise ValueError("dx and the finite state norm must be positive.")
    return psi / np.sqrt(norm)


def gaussian(x, *, x0=0.0, sigma=0.5, p0=0.0, hbar=1.0):
    """Return a normalized Gaussian packet.

    x0: mean position; sigma: position standard deviation;
    p0: mean momentum; hbar: Planck constant in the chosen units.
    Analytic moments assume the packet is well resolved and away from edges.
    """
    if not np.all(np.isfinite([x0, sigma, p0, hbar])):
        raise ValueError("Packet parameters must be finite.")
    if sigma <= 0 or hbar <= 0:
        raise ValueError("sigma and hbar must be positive.")
    x = np.asarray(x, dtype=float)
    dx = grid_spacing(x)
    displacement = x - x0
    envelope = np.exp(-displacement**2 / (4 * sigma**2))
    phase = np.exp(1j * p0 * displacement / hbar)
    return normalize(envelope * phase, dx)
