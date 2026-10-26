"""Newtonian motion using velocity Verlet (momentum form)."""

import numpy as np


def verlet_step(x, p, force, dt, *, mass=1.0):
    """Return updated position and momentum for a static position-only force.

    force is a function: force(x) returns -dV/dx.
    One step = half momentum kick, position drift, half momentum kick.
    Use smaller dt to check convergence. Inputs may be scalars or arrays.
    """
    if not np.isfinite(mass) or mass <= 0 or not np.isfinite(dt):
        raise ValueError("mass must be positive and finite; dt must be finite.")
    x, p = np.asarray(x, dtype=float), np.asarray(p, dtype=float)
    if x.shape != p.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
        raise ValueError("x and p must be finite and have matching shapes.")
    p_half = p + 0.5 * dt * force(x)
    x_next = x + dt * p_half / mass
    p_next = p_half + 0.5 * dt * force(x_next)
    return x_next, p_next

def gaussian_ensemble(*, x0=0.0, p0=0.0, sigma=0.5, hbar=1.0,
                      count=4096, seed=42):
    """Sample independent Gaussian positions and momenta matching our packet.

    Returns two arrays, one position and momentum per independent classical
    particle. This matches the initial uncorrelated Gaussian Wigner distribution;
    Finite samples fluctuate around the intended moments.
    """
    if not np.all(np.isfinite([x0, p0, sigma, hbar])) or sigma <= 0 or hbar <= 0:
        raise ValueError("Parameters must be finite; sigma and hbar positive.")
    if not isinstance(count, (int, np.integer)) or count < 2:
        raise ValueError("count must be an integer of at least two.")
    rng = np.random.default_rng(seed)
    positions = rng.normal(x0, sigma, count)
    momenta = rng.normal(p0, hbar / (2 * sigma), count)
    return positions, momenta

def ensemble_statistics(values):
    """Equal-weight population moments, plus estimated standard error of mean.

    Width describes the distribution; standard error describes uncertainty in
    the sampled mean. It does not include numerical integration error.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Provide at least two finite values in a 1D array.")
    return dict(mean=float(np.mean(values)), variance=float(np.var(values)),
                width=float(np.std(values)),
                mean_se=float(np.std(values, ddof=1)/np.sqrt(values.size)))

