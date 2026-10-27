"""Ideal efficient continuous-position-monitoring approximation.

Strength convention: d<E>/dt = k*hbar**2/m after averaging over outcomes.
Finite dt implements repeated weak Gaussian measurements, not exact continuous
monitoring. Convergence must be checked statistically, not by comparing paths
with the same seed at different dt.
"""
import numpy as np
from wavefunctions import grid_spacing
from evolution import potential_step
from measurement import measure_position


def monitored_step(x, psi, V, dt, *, strength=1.0, mass=1.0, hbar=1.0, rng=None):
    """Return (updated_state, detector_reading_or_None).

    Hamiltonian half-step / Gaussian measurement / Hamiltonian half-step.
    At strength=0 use ordinary unitary evolution and draw no random numbers.
    A detector reading is noisy and is not the conditional mean position.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative.")
    dx = grid_spacing(x)
    if strength == 0:
        return potential_step(psi, V, dx, dt, mass=mass, hbar=hbar), None
    half = potential_step(psi, V, dx, dt/2, mass=mass, hbar=hbar)
    resolution = 1 / np.sqrt(8 * strength * dt)
    reading, conditioned = measure_position(x, half, resolution=resolution, rng=rng)
    return potential_step(conditioned, V, dx, dt/2, mass=mass, hbar=hbar), reading
