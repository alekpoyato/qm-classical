"""Time evolution on a uniform, periodic position grid.

Keep the packet far from the edges: the FFT treats the interval as repeating.
Requires NumPy. Supports H = p^2/(2m) + V(x) with a static real potential.
"""

import numpy as np


def free_step(psi, dx, dt, *, mass=1.0, hbar=1.0):
    """Return a new wavefunction evolved freely for a time dt.

    psi: complex amplitudes at equally spaced positions.
    dx: distance between neighboring positions.
    dt: elapsed time; negative values evolve backward.
    mass, hbar: positive constants in your chosen units.

    The input is not modified or renormalized. The method preserves its norm
    up to floating-point rounding. Free evolution is exact for the discrete
    Fourier modes, but the finite grid still limits physical accuracy.
    """
    psi = np.asarray(psi, dtype=complex)
    if psi.ndim != 1 or psi.size < 2 or not np.all(np.isfinite(psi)):
        raise ValueError("psi must be a finite 1D array with at least two entries.")
    if not np.all(np.isfinite([dx, dt, mass, hbar])):
        raise ValueError("dx, dt, mass, and hbar must be finite.")
    if dx <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError("dx, mass, and hbar must be positive.")

    # Label the Fourier components with their momenta: p = hbar * 2*pi*f.
    frequencies = np.fft.fftfreq(psi.size, d=dx)
    p = 2 * np.pi * hbar * frequencies

    # Each free-particle component has energy p^2/(2m).
    energy = p**2 / (2 * mass)
    time_phase = np.exp(-1j * energy * dt / hbar)

    # Decompose the state, evolve each component, and recombine.
    # Keep raw FFT normalization and ordering: ifft reverses fft directly.
    coefficients = np.fft.fft(psi)
    evolved_coefficients = coefficients * time_phase
    return np.fft.ifft(evolved_coefficients)


def potential_step(psi, V, dx, dt, *, mass=1.0, hbar=1.0):
    """Advance H = p^2/(2m) + V(x) by ONE small split step.

    V is an array of potential energies on the same grid as psi, e.g.
    V = harmonic(x, mass=mass, omega=1.0).
    It must be real; absorbing or other complex potentials are not supported.

    Unlike free_step, this is approximate. Reduce dt and compare results
    to check accuracy. For smooth, resolved potentials the method has
    second-order global time accuracy. Sharp well edges need spatial checks too.
    The input state is not modified or renormalized.
    """
    psi = np.asarray(psi, dtype=complex)
    V = np.asarray(V)
    if psi.ndim != 1 or psi.size < 2 or not np.all(np.isfinite(psi)):
        raise ValueError("psi must be a finite 1D array with at least two entries.")
    if V.shape != psi.shape or np.iscomplexobj(V) or not np.all(np.isfinite(V)):
        raise ValueError("V must be a finite real array with the same shape as psi.")
    if not np.all(np.isfinite([dx, dt, mass, hbar])):
        raise ValueError("dx, dt, mass, and hbar must be finite.")
    if dx <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError("dx, mass, and hbar must be positive.")

    # Potential energy is diagonal in position space: multiply by a phase.
    half_potential_phase = np.exp(-1j * V * (dt / 2) / hbar)
    psi_half = half_potential_phase * psi

    # Kinetic energy is diagonal in momentum space; free_step handles its FFTs.
    psi_kinetic = free_step(psi_half, dx, dt, mass=mass, hbar=hbar)

    # Complete the symmetric potential / kinetic / potential sequence.
    return half_potential_phase * psi_kinetic
