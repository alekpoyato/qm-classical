"""Potential energies evaluated on a position grid.

These functions return V(x), not forces or time-evolution operators.
Use the same units and mass as in the evolution functions.
"""

import numpy as np


def harmonic(x, *, mass=1.0, omega=1.0, center=0.0):
    """Return V(x) = (1/2) * mass * omega**2 * (x - center)**2.

    x: positions at which to evaluate the potential.
    mass: positive particle mass.
    omega: nonnegative angular frequency (radians per unit time).
    center: equilibrium position, where the potential is minimal.

    The corresponding force is -mass * omega**2 * (x - center).
    omega=0 gives zero potential everywhere.
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Positions must be finite.")
    if not np.all(np.isfinite([mass, omega, center])):
        raise ValueError("mass, omega, and center must be finite.")
    if mass <= 0 or omega < 0:
        raise ValueError("mass must be positive and omega nonnegative.")

    displacement = x - center
    return 0.5 * mass * omega**2 * displacement**2

def square_well(x, *, width=4.0, height=10.0, center=0.0):
    """Return a finite square well: zero inside, height outside."""
    x = np.asarray(x, dtype=float)

    if not np.all(np.isfinite(x)):
        raise ValueError("Positions must be finite.")
    if not np.all(np.isfinite([width, height, center])):
        raise ValueError("Parameters must be finite.")
    if width <= 0 or height < 0:
        raise ValueError("width must be positive and height nonnegative.")

    inside = np.abs(x - center) < width / 2
    return np.where(inside, 0.0, height)

def quartic(x, *, mass=1.0, omega=1.0, lam=0.1, center=0.0):
    """Harmonic plus quartic potential; lam >= 0 keeps it confining."""
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and nonnegative.")
    x = np.asarray(x, dtype=float)
    return harmonic(x, mass=mass, omega=omega, center=center) + lam * (x-center)**4


def quartic_force(x, *, mass=1.0, omega=1.0, lam=0.1, center=0.0):
    """Force -dV/dx for the same parameters as quartic()."""
    # Reuse potential validation to keep both definitions consistent.
    quartic(x, mass=mass, omega=omega, lam=lam, center=center)
    displacement = np.asarray(x, dtype=float) - center
    return -mass * omega**2 * displacement - 4 * lam * displacement**3
