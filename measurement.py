"""An ideal, finite-resolution Gaussian position measurement.

This is one specified measurement model, not every possible position detector.
The result y has density equal to |psi|^2 convolved with a Gaussian of width r.
Conditioning multiplies amplitudes by exp(-(x-y)^2/(4*r^2)), retaining phase.
No separate momentum kick is added: backaction is already in the state update.
"""

import numpy as np
from wavefunctions import grid_spacing, normalize


def measure_position(x, psi, *, resolution=0.5, rng=None):
    """Return (outcome, posterior_state) without changing the input.

    resolution is the detector's position standard deviation r, not the final
    packet width. Supply a numpy.random.Generator for reproducible draws.
    Use enough spatial resolution to resolve the resulting packet.
    """
    x = np.asarray(x, dtype=float)
    psi = np.asarray(psi, dtype=complex)
    dx = grid_spacing(x)
    if psi.shape != x.shape or not np.all(np.isfinite(psi)):
        raise ValueError("psi must be finite and match the position grid.")
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be positive and finite.")
    density = np.abs(psi)**2
    total = density.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("State must have finite, positive norm.")
    if rng is None:
        rng = np.random.default_rng()

    # Sample the Born distribution convolved with detector noise. The sampled
    # grid index is only a sampling device, not a hidden particle position.
    index = rng.choice(x.size, p=density/total)
    outcome = float(x[index] + rng.normal(0, resolution))

    # Evaluate the Gaussian update in log magnitude to avoid underflow.
    nonzero = np.abs(psi) > 0
    log_magnitude = np.full(x.shape, -np.inf)
    log_magnitude[nonzero] = (np.log(np.abs(psi[nonzero]))
                             - ((x[nonzero]-outcome)/resolution)**2/4)
    log_magnitude -= np.max(log_magnitude)
    posterior = np.exp(log_magnitude) * np.exp(1j*np.angle(psi))
    return outcome, normalize(posterior, dx)
