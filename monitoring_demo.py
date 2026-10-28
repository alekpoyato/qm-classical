"""Animate a quantum anharmonic oscillator and compare it with classical motion.

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
from potentials import quartic, quartic_force
from classical import verlet_step, gaussian_ensemble, ensemble_statistics
from monitoring import monitored_step


def main(show=True, *, count=2048, seed=42, dt=0.001):
    if not np.isfinite(dt) or dt <= 0 or dt > .01:
        raise ValueError("Use 0 < dt <= 0.01 and compare smaller steps.")
    x = position_grid(-40, 40, 2048)
    dx = x[1] - x[0]
    hbar = 1.0
    state = {"psi": None, "V": None, "time": 0.0, "paused": False, "monitoring": True, "blocked": False}

    fig, (ax_x, ax_p) = plt.subplots(1, 2, figsize=(14, 10))
    fig.subplots_adjust(left=.08, right=.95, top=.74, bottom=.52, wspace=.3)
    fig.suptitle("Continuous position monitoring (weak-measurement approximation)", fontsize=17)
    measurement_status = fig.text(.08, .89, "No measurement yet.", fontsize=10)
    status = fig.text(.08, .79, "", fontsize=10)
    for ax, title, xlabel, ylabel in [
        (ax_x, "Position distribution", "Position x", "Probability density |ψ|²"),
        (ax_p, "Momentum distribution", "Momentum p", "Probability density |φ|²"),
    ]:
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.2)

    line_x, = ax_x.plot([], [], lw=2, color="#2768c7", label="Quantum density")
    line_p, = ax_p.plot([], [], lw=2, color="#7b42b5", label="Quantum density")
    classical_x = ax_x.axvline(3, color="#e77722", ls="--", label="Single classical particle")
    classical_p = ax_p.axvline(0, color="#e77722", ls="--", label="Single classical particle")
    quantum_x = ax_x.axvline(3, color="#169c89", ls=":", label="Quantum mean")
    quantum_p = ax_p.axvline(0, color="#169c89", ls=":", label="Quantum mean")
    ensemble_x, = ax_x.plot([], [], color="#c64a91", drawstyle="steps-mid", alpha=.8,
                            label="Unmeasured ensemble")
    ensemble_p, = ax_p.plot([], [], color="#c64a91", drawstyle="steps-mid", alpha=.8,
                            label="Unmeasured ensemble")
    ensemble_mean_x = ax_x.axvline(0, color="#c64a91", ls="--", label="Ensemble mean")
    ensemble_mean_p = ax_p.axvline(0, color="#c64a91", ls="--", label="Ensemble mean")
    reference_x, = ax_x.plot([], [], color="black", ls="--", alpha=.65, label="Unmeasured quantum")
    reference_p, = ax_p.plot([], [], color="black", ls="--", alpha=.65, label="Unmeasured quantum")
    ax_x.legend(loc="upper right", fontsize=7)
    ax_p.legend(loc="upper right", fontsize=7)
    fig.text(.08, .39,
             "Preparation and strength sliders restart. Monitoring switch acts immediately; Pause freezes time.\n"
             f"Pink histograms: {count:,} independent classical particles, seed {seed}. Slider resets reuse the seed.\n"
             "One random trajectory: heating prediction is an ensemble average, not a target for this path.", fontsize=10)

    def slider(y, label, low, high, initial):
        return Slider(fig.add_axes([.24, y, .64, .024]), label, low, high, valinit=initial)

    controls = {
        "x0": slider(.34, "Initial position", -3, 3, 1.5),
        "p0": slider(.295, "Initial momentum", -3, 3, 0),
        "sigma": slider(.25, "Initial width", .3, 1.2, .5),
        "mass": slider(.205, "Mass", .5, 2, 1),
        "omega": slider(.16, "Angular frequency ω", .5, 2, 1),
        "lam": slider(.115, "Quartic strength λ", 0, .3, .1),
    }
    strength = slider(.075, "Monitoring strength k", 0, 2, .5)
    speed = Slider(fig.add_axes([.24, .025, .25, .024]), "Playback speed", .25, 2, valinit=1)
    pause = Button(fig.add_axes([.54, .01, .10, .045]), "Pause")
    reset = Button(fig.add_axes([.66, .01, .10, .045]), "Reset")
    preset = Button(fig.add_axes([.78, .01, .17, .045]), "Monitoring: ON")

    def draw_frame():
        t, psi = state["time"], state["psi"]
        m, w = controls["mass"].val, controls["omega"].val
        x0, p0 = controls["x0"].val, controls["p0"].val
        # The classical reference is integrated numerically with the same dt.
        xc, pc = state["xc"], state["pc"]
        rho = probability_density(psi)
        p, phi = momentum_wavefunction(psi, dx, hbar, x_min=x[0])
        density_p = probability_density(phi)
        ref_rho = probability_density(state["reference"])
        _, ref_phi = momentum_wavefunction(state["reference"], dx, hbar, x_min=x[0])
        ref_density_p = probability_density(ref_phi)
        reference_x.set_data(x, ref_rho)
        reference_p.set_data(p, ref_density_p)
        line_x.set_data(x, rho)
        line_p.set_data(p, density_p)
        classical_x.set_xdata([xc, xc])
        classical_p.set_xdata([pc, pc])
        stats = position_statistics(x, psi)
        dp = p[1] - p[0]
        norm_p = np.sum(density_p) * dp
        mean_p = np.sum(p * density_p) * dp / norm_p
        quantum_x.set_xdata([stats["mean"]]*2)
        quantum_p.set_xdata([mean_p]*2)
        width_p = np.sqrt(np.sum((p - mean_p)**2 * density_p) * dp / norm_p)
        ex = ensemble_statistics(state["ensemble_x"])
        ep = ensemble_statistics(state["ensemble_p"])
        ensemble_mean_x.set_xdata([ex["mean"]]*2)
        ensemble_mean_p.set_xdata([ep["mean"]]*2)
        # Counts / (ALL particles * bin width): do not renormalize visible bins.
        for axis, values, curve in [(ax_x, state["ensemble_x"], ensemble_x),
                                    (ax_p, state["ensemble_p"], ensemble_p)]:
            lo, hi = axis.get_xlim()
            lo, hi = min(lo, float(values.min())*1.1), max(hi, float(values.max())*1.1)
            axis.set_xlim(lo, hi)
            bins = np.linspace(lo, hi, 101)
            counts, edges = np.histogram(values, bins=bins)
            density = counts / (values.size * np.diff(edges))
            curve.set_data((edges[:-1]+edges[1:])/2, density)
            axis.set_ylim(0, max(axis.get_ylim()[1], 1.1*density.max()))
        energy = (np.sum(p**2 / (2*m) * density_p) * dp / norm_p
                  + np.sum(state["V"] * rho) * dx / stats["norm"])
        force_mean = np.sum(state["force"](x) * rho) * dx / stats["norm"]
        force_gap = force_mean - state["force"](stats["mean"])
        edge = np.sum(rho[np.abs(x) > 32]) * dx / stats["norm"]
        tail = np.sum(density_p[np.abs(p) > .8*np.max(np.abs(p))]) * dp / norm_p
        state["diagnostics"] = dict(energy=energy, norm=stats["norm"], edge=edge,
                                    tail=tail, mean_x=stats["mean"], mean_p=mean_p,
                                    width_x=stats["width"], width_p=width_p, ensemble_x=ex, ensemble_p=ep)
        status.set_text(
            f"t = {t:.2f}     ⟨x⟩ = {stats['mean']:.3f}     ⟨p⟩ = {mean_p:.3f}     "
            f"σx = {stats['width']:.3f}     σp = {width_p:.3f}\n"
            f"Probability = {stats['norm']:.6f}     Total energy = {energy:.5f}     "
            f"|⟨x⟩ − classical x| = {abs(stats['mean'] - xc):.2e}\n"
            f"⟨F⟩ − F(⟨x⟩) = {force_gap:.3f}     Edge probability = {edge:.1e}     "
            f"High-momentum tail = {tail:.1e}\n"
            f"Ensemble: ⟨x⟩ = {ex['mean']:.3f} ± {ex['mean_se']:.3f} SE, "
            f"σx = {ex['width']:.3f};  ⟨p⟩ = {ep['mean']:.3f} ± {ep['mean_se']:.3f} SE, "
            f"σp = {ep['width']:.3f}")
        reference_stats = position_statistics(x, state["reference"])
        ref_energy = (np.sum(p**2/(2*m)*ref_density_p)*dp + np.sum(state["V"]*ref_rho)*dx) / reference_stats["norm"]
        measurement_status.set_text(
            f"Monitoring {'ON' if state['monitoring'] else 'OFF'} | k = {strength.val:.2f} | dt = {dt:g} | seed = {seed}\n"
            f"σx monitored / unmeasured: {stats['width']:.3f} / {reference_stats['width']:.3f}    "
            f"E monitored − unmeasured = {energy-ref_energy:+.3f}\n"
            f"Expected average measurement heating so far: {state['heating']:.3f}")
        if edge > 1e-5 or tail > 1e-5:
            state['paused'] = True
            state['blocked'] = True
            pause.label.set_text('Reset needed')
            measurement_status.set_text(measurement_status.get_text() + '\nGrid limit reached: reset; do not interpret this run.')
        # Expand the view if nonlinear dynamics creates taller peaks or wider tails.
        ax_x.set_ylim(0, max(ax_x.get_ylim()[1], 1.1*max(np.max(rho),np.max(ref_rho))))
        ax_p.set_ylim(0, max(ax_p.get_ylim()[1], 1.1*max(np.max(density_p),np.max(ref_density_p))))
        occupied = np.abs(p[density_p > np.max(density_p)*1e-5])
        if occupied.size and occupied.max() > ax_p.get_xlim()[1]:
            ax_p.set_xlim(-1.1*occupied.max(), 1.1*occupied.max())
        return line_x, line_p, classical_x, classical_p, quantum_x, quantum_p, ensemble_x, ensemble_p, ensemble_mean_x, ensemble_mean_p, reference_x, reference_p, status

    def prepare(_=None):
        m, w = controls["mass"].val, controls["omega"].val
        sigma = controls["sigma"].val
        x0, p0 = controls["x0"].val, controls["p0"].val
        state["psi"] = gaussian(x, x0=x0, p0=p0, sigma=sigma, hbar=hbar)
        lam = controls["lam"].val
        state["V"] = quartic(x, mass=m, omega=w, lam=lam)
        state["force"] = lambda position: quartic_force(position, mass=m, omega=w, lam=lam)
        state["xc"], state["pc"] = x0, p0
        state["ensemble_x"], state["ensemble_p"] = gaussian_ensemble(
            x0=x0, p0=p0, sigma=sigma, hbar=hbar, count=count, seed=seed)
        state["rng"] = np.random.default_rng(seed + 1)
        state["reference"] = state["psi"].copy()
        state["heating"] = 0.0
        state["blocked"] = False
        pause.label.set_text("Play" if state["paused"] else "Pause")
        measurement_status.set_text("No measurement yet. Detector resolution changes do not restart time.")
        state["time"] = 0.0
        # Use generous display limits; these are not the simulation boundaries.
        amplitude_x = np.hypot(x0, p0/(m*w))
        widths_x = [sigma, hbar/(2*m*w*sigma)]
        widths_p = [hbar/(2*sigma), m*w*sigma]
        limit_x = min(35, max(6, amplitude_x + 5*max(widths_x)))
        limit_p = max(12, m*w*amplitude_x + 6*max(widths_p))
        ax_x.set(xlim=(-limit_x, limit_x), ylim=(0, 1.15/(np.sqrt(2*np.pi)*min(widths_x))))
        ax_p.set(xlim=(-limit_p, limit_p), ylim=(0, 1.15/(np.sqrt(2*np.pi)*min(widths_p))))
        draw_frame()
        fig.canvas.draw_idle()

    def animate(_):
        if not state["paused"]:
            m, w = controls["mass"].val, controls["omega"].val
            # Advance the CURRENT state with small Strang steps, not one large step.
            for _ in range(max(1, round(.015 * speed.val / dt))):
                step = min(dt, 8.0 - state["time"])
                if step <= 1e-12:
                    state["paused"] = True
                    pause.label.set_text("Play")
                    break
                k = strength.val if state["monitoring"] else 0.0
                state["psi"], state["reading"] = monitored_step(
                    x, state["psi"], state["V"], step, strength=k,
                    mass=m, hbar=hbar, rng=state["rng"])
                state["reference"] = potential_step(state["reference"], state["V"], dx, step,
                                                     mass=m, hbar=hbar)
                state["heating"] += k*hbar**2*step/m
                state["xc"], state["pc"] = verlet_step(
                    state["xc"], state["pc"], state["force"], step, mass=m)
                state["ensemble_x"], state["ensemble_p"] = verlet_step(
                    state["ensemble_x"], state["ensemble_p"], state["force"], step, mass=m)
                state["time"] += step
        return draw_frame()

    def toggle(_):
        if state["blocked"]:
            return
        if state["time"] >= 8.0 - 1e-10:
            prepare()
        state["paused"] = not state["paused"]
        pause.label.set_text("Play" if state["paused"] else "Pause")
        fig.canvas.draw_idle()

    def restart(_):
        state["paused"] = False
        pause.label.set_text("Pause")
        prepare()

    def toggle_monitoring(_):
        state["monitoring"] = not state["monitoring"]
        preset.label.set_text("Monitoring: ON" if state["monitoring"] else "Monitoring: OFF")
        draw_frame()
        fig.canvas.draw_idle()

    for control in controls.values():
        control.on_changed(prepare)
    strength.on_changed(prepare)
    pause.on_clicked(toggle)
    reset.on_clicked(restart)
    preset.on_clicked(toggle_monitoring)
    prepare()
    animation = FuncAnimation(fig, animate, interval=30, blit=False, cache_frame_data=False)
    # Retain widgets and animation for as long as the figure exists.
    fig._demo = dict(animation=animation, controls=controls, buttons=(pause, reset, preset),
                     speed=speed, state=state, advance=animate, x=x,
                     monitor=toggle_monitoring, strength=strength)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=.001)
    args = parser.parse_args()
    main(count=args.count, seed=args.seed, dt=args.dt)
