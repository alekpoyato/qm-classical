"""Animate position and momentum distributions with live sliders.

Requires NumPy, Matplotlib, and the three companion modules."""

import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import numpy as np

from evolution import free_step
from wavefunctions import gaussian, position_grid, probability_density
from observables import momentum_wavefunction

def main():
    x = position_grid(-100.0, 100.0, 8192)
    dx = x[1] - x[0]
    hbar = 1.0
    state = {"time": 0.0, "paused": False, "psi": None}

    fig, (ax, ax_p) = plt.subplots(1, 2, figsize=(13, 8))

    fig.subplots_adjust(
        left=0.08, right=0.96,
        top=0.86, bottom=0.43,
        wspace=0.30,
    )
    fig.suptitle("Free quantum wave packet", fontsize=17)

    # Position distribution
    ax.set(
        title="Position",
        xlabel="Position x",
        ylabel="Probability density |ψ(x, t)|²",
        xlim=(-25, 25),
        ylim=(0, 0.9),
    )
    ax.grid(alpha=0.2)

    line, = ax.plot([], [], color="#2768c7", linewidth=2)
    center = ax.axvline(
        0, color="#d66a22", linestyle="--",
        label="Mean position",
    )
    ax.legend(loc="upper right")

    # Momentum distribution
    ax_p.set(
        title="Momentum",
        xlabel="Momentum p",
        ylabel="Probability density |φ(p, t)|²",
    )
    ax_p.grid(alpha=0.2)

    momentum_line, = ax_p.plot(
        [], [], color="#7b42b5", linewidth=2,
    )
    momentum_center = ax_p.axvline(
        0, color="#d66a22", linestyle="--",
        label="Mean momentum",
    )
    ax_p.legend(loc="upper right")
    status = fig.text(0.12, 0.91, "", fontsize=10)
    fig.text(0.12, 0.365,
             "Changing a preparation slider restarts time. Drag Time to inspect a frame.\n"
             "The animation stops at t = 4; press Reset to replay. Units are dimensionless, ℏ = 1.",
             fontsize=10)

    def slider(y, label, low, high, initial, step=None):
        return Slider(fig.add_axes([0.24, y, 0.65, 0.025]), label, low, high,
                      valinit=initial, valstep=step)

    controls = {
        "x0": slider(0.31, "Initial position", -5, 5, 0),
        "p0": slider(0.265, "Mean momentum", -4, 4, 2),
        "sigma": slider(0.22, "Initial width", 0.3, 2, 0.5),
        "mass": slider(0.175, "Mass", 0.5, 3, 1),
    }
    time_slider = slider(0.115, "Time", 0, 4, 0)
    speed_slider = Slider(fig.add_axes([0.24, 0.065, 0.34, 0.025]),
                          "Playback speed", 0.1, 2, valinit=0.5)
    pause = Button(fig.add_axes([0.65, 0.045, 0.13, 0.045]), "Pause")
    reset = Button(fig.add_axes([0.81, 0.045, 0.13, 0.045]), "Reset")

    def draw_frame():
        t = state["time"]
        mass = controls["mass"].val
        psi_t = free_step(state["psi"], dx, t, mass=mass, hbar=hbar)
        line.set_data(x, probability_density(psi_t))

        # Fourier transform the evolved state to obtain its momentum density.
        p, phi = momentum_wavefunction(psi_t, dx, hbar=hbar, x_min=x[0])
        momentum_density = probability_density(phi)
        momentum_line.set_data(p, momentum_density)
        mean_p = controls["p0"].val
        momentum_width = hbar / (2 * controls["sigma"].val)
        momentum_center.set_xdata([mean_p, mean_p])
        ax_p.set_xlim(mean_p - 4 * momentum_width, mean_p + 4 * momentum_width)
        ax_p.set_ylim(0, 1.12 / (np.sqrt(2 * np.pi) * momentum_width))
        mean = controls["x0"].val + controls["p0"].val * t / mass
        sigma = controls["sigma"].val
        width = np.sqrt(sigma**2 + (hbar * t / (2 * mass * sigma))**2)
        center.set_xdata([mean, mean])
        # Keep essentially the entire packet visible, including broad packets.
        half_view = max(12, 4 * width)
        ax.set_xlim(mean - half_view, mean + half_view)
        ax.set_ylim(0, 1.12 / (np.sqrt(2 * np.pi) * sigma))
        norm = np.sum(probability_density(psi_t)) * dx
        status.set_text(f"t = {t:.2f}     Mean position = {mean:.2f}     "
                        f"σx = {width:.2f}     σp = {momentum_width:.2f}     "
                        f"σxσp = {width * momentum_width:.2f}     Probability = {norm:.6f}")

    def set_time(value):
        state["time"] = float(value)
        # Avoid triggering the user-scrubbing callback during playback.
        time_slider.eventson = False
        time_slider.set_val(value)
        time_slider.eventson = True

    def prepare(_=None):
        state["psi"] = gaussian(x, x0=controls["x0"].val,
                                sigma=controls["sigma"].val,
                                p0=controls["p0"].val, hbar=hbar)
        set_time(0)
        draw_frame()
        fig.canvas.draw_idle()

    def scrub(value):
        state["paused"] = True
        pause.label.set_text("Play")
        state["time"] = float(value)
        draw_frame()
        fig.canvas.draw_idle()

    def toggle(_):
        if state["time"] >= 4:
            set_time(0)
        state["paused"] = not state["paused"]
        pause.label.set_text("Play" if state["paused"] else "Pause")

    def restart(_):
        state["paused"] = False
        pause.label.set_text("Pause")
        prepare()

    def animate(_):
        if not state["paused"]:
            set_time(min(4, state["time"] + 0.03 * speed_slider.val))
            if state["time"] >= 4:
                state["paused"] = True
                pause.label.set_text("Play")
            draw_frame()
        return line, center, momentum_line, momentum_center, status

    for control in controls.values():
        control.on_changed(prepare)
    time_slider.on_changed(scrub)
    pause.on_clicked(toggle)
    reset.on_clicked(restart)
    prepare()
    # Retain a reference so the animation remains alive while the window is open.
    animation = FuncAnimation(fig, animate, interval=30, blit=False,
                              cache_frame_data=False)
    plt.show()
    return animation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()

