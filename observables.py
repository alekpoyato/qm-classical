"""Calculate predictions from a sampled 1D wavefunction without changing it.

The position grid must be uniform. FFTs assume a periodic interval: keep the 
packet away from the edges and use enough points to resolve its spatial oscillations.
"""

import numpy as np
from wavefunctions import grid_spacing, probability_density


def _check_state(x, psi):
    """Validate the grid and state; do not silently normalize the state."""
    x = np.asarray(x, dtype=float)
    psi = np.asarray(psi, dtype=complex)
    dx = grid_spacing(x)
    if psi.shape != x.shape or not np.all(np.isfinite(psi)):
        raise ValueError("psi must be finite and have the same shape as x.")
    norm = np.sum(probability_density(psi)) * dx
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("The state must have a finite, positive norm.")
    return x, psi, dx


def _statistics(coordinate, density, spacing):
    """Integrate a density to obtain its norm, mean, variance, and width."""
    norm = np.sum(density) * spacing
    mean = np.sum(coordinate * density) * spacing / norm
    variance = np.sum((coordinate - mean)**2 * density) * spacing / norm
    return {
        "norm": float(norm),
        "mean": float(mean),
        "variance": float(variance),
        "width": float(np.sqrt(variance)),
    }


def position_statistics(x, psi):
    """Return position norm, mean, variance, and standard deviation (width).

    Means divide by the measured norm, while norm is reported separately.
    """
    x, psi, dx = _check_state(x, psi)
    return _statistics(x, probability_density(psi), dx)


def momentum_wavefunction(psi, dx, hbar=1.0, *, x_min=0.0):
    """Return ascending momenta p and their complex amplitudes phi.

    Assumes positions x_min + j*dx. Pass x_min=x[0] for the physical
    position origin; omitting it changes phases but not probabilities.
    For a normalized input, sum(|phi|^2)*dp = 1.
    """
    psi = np.asarray(psi, dtype=complex)
    if psi.ndim != 1 or psi.size < 2 or not np.all(np.isfinite(psi)):
        raise ValueError("psi must be a finite 1D array with at least two entries.")
    if not np.all(np.isfinite([dx, hbar, x_min])) or dx <= 0 or hbar <= 0:
        raise ValueError("dx and hbar must be positive; all parameters finite.")

    # FFT frequencies are in cycles per unit length. p = hbar * 2*pi*f.
    frequencies = np.fft.fftfreq(psi.size, d=dx)
    p = 2 * np.pi * hbar * frequencies

    # Approximate phi(p) = integral psi(x)*exp(-i*p*x/hbar) dx / sqrt(2*pi*hbar).
    phi = np.fft.fft(psi) * dx / np.sqrt(2 * np.pi * hbar)
    phi = phi * np.exp(-1j * p * x_min / hbar)

    # Reorder BOTH arrays together, from negative to positive momentum.
    return np.fft.fftshift(p), np.fft.fftshift(phi)


def momentum_statistics(x, psi, hbar=1.0):
    """Return momentum norm, mean, variance, and standard deviation."""
    x, psi, dx = _check_state(x, psi)
    p, phi = momentum_wavefunction(psi, dx, hbar, x_min=x[0])
    dp = p[1] - p[0]
    return _statistics(p, probability_density(phi), dp)


def mean_kinetic_energy(x, psi, mass=1.0, hbar=1.0):
    """Return <p^2/(2m)>; this is total energy only for a free particle V=0."""
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("mass must be finite and positive.")
    stats = momentum_statistics(x, psi, hbar)
    # <p^2> = Var(p) + <p>^2.
    return (stats["variance"] + stats["mean"]**2) / (2 * mass)
