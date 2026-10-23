"""Animate a quantum harmonic oscillator and compare it with classical motion.

Requires NumPy, Matplotlib, wavefunctions.py, observables.py, potentials.py,
plus evolution.py. Preparation sliders restart the experiment at t=0.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

from wavefunctions import position_grid, gaussian, probability_density
from observables import momentum_wavefunction, position_statistics
from evolution import potential_step
from potentials import harmonic


def main(show=True):
    x = position_grid(-40, 40, 2048)
    dx = x[1] - x[0]
    hbar = 1.0
    dt = 0.002  # Numerical time step; playback speed changes step count only.
    state = {"psi": None, "V": None, "time": 0.0, "paused": False}

    fig, (ax_x, ax_p) = plt.subplots(1, 2, figsize=(13, 8))
    fig.subplots_adjust(left=.08, right=.95, top=.83, bottom=.50, wspace=.3)
    fig.suptitle("Quantum packet in a harmonic potential", fontsize=17)
    status = fig.text(.08, .90, "", fontsize=10)
    for ax, title, xlabel, ylabel in [
        (ax_x, "Position distribution", "Position x", "Probability density |ψ|²"),
        (ax_p, "Momentum distribution", "Momentum p", "Probability density |φ|²"),
    ]:
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.2)

    line_x, = ax_x.plot([], [], lw=2, color="#2768c7", label="Quantum density")
    line_p, = ax_p.plot([], [], lw=2, color="#7b42b5", label="Quantum density")
    classical_x = ax_x.axvline(3, color="#e77722", ls="--", label="Classical position")
    classical_p = ax_p.axvline(0, color="#e77722", ls="--", label="Classical momentum")
    ax_x.legend(loc="upper right", fontsize=9)
    ax_p.legend(loc="upper right", fontsize=9)
    fig.text(.08, .365,
             "Changing a preparation slider restarts time. Fixed-width preset chooses σ = √(ℏ / 2mω).\n"
             "Orange markers follow Newton's equation. Evolution stops after two oscillator periods.\n"
             "Units are dimensionless; ℏ = 1. Plot axes stay fixed during each run.", fontsize=10)

    def slider(y, label, low, high, initial):
        return Slider(fig.add_axes([.24, y, .64, .024]), label, low, high, valinit=initial)

    controls = {
        "x0": slider(.30, "Initial position", -4, 4, 3),
        "p0": slider(.255, "Initial momentum", -3, 3, 0),
        "sigma": slider(.21, "Initial width", .25, 2, 1 / np.sqrt(2)),
        "mass": slider(.165, "Mass", .5, 2, 1),
        "omega": slider(.12, "Angular frequency ω", .5, 2, 1),
    }
    speed = Slider(fig.add_axes([.24, .065, .25, .024]), "Playback speed", .25, 2, valinit=1)
    pause = Button(fig.add_axes([.54, .045, .10, .05]), "Pause")
    reset = Button(fig.add_axes([.66, .045, .10, .05]), "Reset")
    preset = Button(fig.add_axes([.78, .045, .17, .05]), "Fixed-width preset")

    def draw_frame():
        t, psi = state["time"], state["psi"]
        m, w = controls["mass"].val, controls["omega"].val
        x0, p0 = controls["x0"].val, controls["p0"].val
        # Classical solution with the same initial means and equilibrium at zero.
        xc = x0 * np.cos(w*t) + p0 / (m*w) * np.sin(w*t)
        pc = p0 * np.cos(w*t) - m*w*x0 * np.sin(w*t)
        rho = probability_density(psi)
        p, phi = momentum_wavefunction(psi, dx, hbar, x_min=x[0])
        density_p = probability_density(phi)
        line_x.set_data(x, rho)
        line_p.set_data(p, density_p)
        classical_x.set_xdata([xc, xc])
        classical_p.set_xdata([pc, pc])
        stats = position_statistics(x, psi)
        dp = p[1] - p[0]
        norm_p = np.sum(density_p) * dp
        mean_p = np.sum(p * density_p) * dp / norm_p
        width_p = np.sqrt(np.sum((p - mean_p)**2 * density_p) * dp / norm_p)
        energy = (np.sum(p**2 / (2*m) * density_p) * dp / norm_p
                  + np.sum(state["V"] * rho) * dx / stats["norm"])
        status.set_text(
            f"t = {t:.2f}     ⟨x⟩ = {stats['mean']:.3f}     ⟨p⟩ = {mean_p:.3f}     "
            f"σx = {stats['width']:.3f}     σp = {width_p:.3f}\n"
            f"Probability = {stats['norm']:.6f}     Total energy = {energy:.5f}     "
            f"|⟨x⟩ − classical x| = {abs(stats['mean'] - xc):.2e}")
        return line_x, line_p, classical_x, classical_p, status

    def prepare(_=None):
        m, w = controls["mass"].val, controls["omega"].val
        sigma = controls["sigma"].val
        x0, p0 = controls["x0"].val, controls["p0"].val
        state["psi"] = gaussian(x, x0=x0, p0=p0, sigma=sigma, hbar=hbar)
        state["V"] = harmonic(x, mass=m, omega=w)
        state["time"] = 0.0
        # Bound the entire classical orbit and Gaussian breathing cycle.
        amplitude_x = np.hypot(x0, p0/(m*w))
        widths_x = [sigma, hbar/(2*m*w*sigma)]
        widths_p = [hbar/(2*sigma), m*w*sigma]
        limit_x = max(5, amplitude_x + 4*max(widths_x))
        limit_p = max(5, m*w*amplitude_x + 4*max(widths_p))
        ax_x.set(xlim=(-limit_x, limit_x), ylim=(0, 1.15/(np.sqrt(2*np.pi)*min(widths_x))))
        ax_p.set(xlim=(-limit_p, limit_p), ylim=(0, 1.15/(np.sqrt(2*np.pi)*min(widths_p))))
        draw_frame()
        fig.canvas.draw_idle()

    def animate(_):
        if not state["paused"]:
            m, w = controls["mass"].val, controls["omega"].val
            # Advance the CURRENT state with small Strang steps, not one large step.
            for _ in range(max(1, round(15 * speed.val))):
                step = min(dt, 4*np.pi/w - state["time"])
                if step <= 1e-12:
                    state["paused"] = True
                    pause.label.set_text("Play")
                    break
                state["psi"] = potential_step(state["psi"], state["V"], dx, step,
                                              mass=m, hbar=hbar)
                state["time"] += step
        return draw_frame()

    def toggle(_):
        if state["time"] >= 4*np.pi/controls["omega"].val - 1e-10:
            prepare()
        state["paused"] = not state["paused"]
        pause.label.set_text("Play" if state["paused"] else "Pause")
        fig.canvas.draw_idle()

    def restart(_):
        state["paused"] = False
        pause.label.set_text("Pause")
        prepare()

    def fixed_width(_):
        m, w = controls["mass"].val, controls["omega"].val
        controls["sigma"].set_val(np.sqrt(hbar/(2*m*w)))

    for control in controls.values():
        control.on_changed(prepare)
    pause.on_clicked(toggle)
    reset.on_clicked(restart)
    preset.on_clicked(fixed_width)
    prepare()
    animation = FuncAnimation(fig, animate, interval=30, blit=False, cache_frame_data=False)
    # Retain widgets and animation for as long as the figure exists.
    fig._demo = dict(animation=animation, controls=controls, buttons=(pause, reset, preset),
                     speed=speed, state=state, advance=animate, x=x)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    main()
