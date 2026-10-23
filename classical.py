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
