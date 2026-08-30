"""Compare three one-step rules on the harmonic oscillator."""

from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from scripts.ode_systems import harmonic_oscillator
from scripts.one_step_methods import (
    explicit_euler_step,
    integrate_fixed_step,
)
from scripts.plotting import (
    CYAN,
    LIGHT_TEXT,
    MUTED_TEXT,
    PINK,
    RED,
    REFERENCE_COLOR,
    configure_plots,
    save_figure,
)
from scripts.reference_solver import reference_solution


FIGURES = Path(__file__).resolve().parent / "figures"
HARMONIC_MATRIX = np.array([[0.0, 1.0], [-1.0, 0.0]])
IDENTITY = np.eye(2)


def harmonic_implicit_euler_step(_rhs, _t, x, h, _args=()):
    return np.linalg.solve(IDENTITY - h * HARMONIC_MATRIX, x)


def harmonic_implicit_midpoint_step(_rhs, _t, x, h, _args=()):
    left = IDENTITY - 0.5 * h * HARMONIC_MATRIX
    right = (IDENTITY + 0.5 * h * HARMONIC_MATRIX) @ x
    return np.linalg.solve(left, right)


METHODS = {
    "explicit Euler": (explicit_euler_step, RED, "--"),
    "implicit Euler": (harmonic_implicit_euler_step, CYAN, "-."),
    "implicit midpoint": (harmonic_implicit_midpoint_step, PINK, ":"),
}


def solve_all(rhs, x0, t_final, h, *, args=()):
    outputs = {}
    timings = {}
    for name, (step, _, _) in METHODS.items():
        start = perf_counter()
        times, states = integrate_fixed_step(step, rhs, x0, t_final, h, args=args)
        timings[name] = perf_counter() - start
        outputs[name] = states
    reference = reference_solution(rhs, x0, times, args=args)
    return times, reference, outputs, timings


def plot_phase_curves(ax, reference, outputs, x0):
    ax.plot(
        reference[:, 0],
        reference[:, 1],
        color=REFERENCE_COLOR,
        linewidth=1.6,
        label="reference",
    )
    for name, states in outputs.items():
        _, color, linestyle = METHODS[name]
        ax.plot(states[:, 0], states[:, 1], color=color, linestyle=linestyle, label=name)
    ax.scatter([x0[0]], [x0[1]], color=REFERENCE_COLOR, s=11, zorder=4)


def add_figure_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=2,
        fontsize=7.2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0.0,
    )


def add_timing_table(ax, timings):
    ax.axis("off")
    labels = {
        "explicit Euler": "explicit",
        "implicit Euler": "implicit",
        "implicit midpoint": "midpoint",
    }
    y_values = [0.62, 0.48, 0.34]
    ax.text(
        0.0,
        0.78,
        "time",
        color=LIGHT_TEXT,
        fontsize=7.0,
        ha="left",
        va="center",
        transform=ax.transAxes,
    )
    ax.text(
        1.0,
        0.78,
        "ms",
        color=LIGHT_TEXT,
        fontsize=7.0,
        ha="right",
        va="center",
        transform=ax.transAxes,
    )
    ax.plot(
        [0.0, 1.0],
        [0.70, 0.70],
        color=MUTED_TEXT,
        linewidth=0.45,
        transform=ax.transAxes,
    )

    for y, name in zip(y_values, METHODS):
        _, color, _ = METHODS[name]
        ax.text(
            0.0,
            y,
            labels[name],
            color=color,
            fontsize=6.5,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )
        ax.text(
            1.0,
            y,
            f"{1000.0 * timings[name]:.2f}",
            color=LIGHT_TEXT,
            fontsize=6.5,
            ha="right",
            va="center",
            transform=ax.transAxes,
        )


def plot_harmonic_phase():
    x0 = np.array([1.0, 0.0])
    times, reference, outputs, timings = solve_all(harmonic_oscillator, x0, 20.0, 0.1)
    fig = plt.figure(figsize=(3.75, 2.65))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.15], wspace=0.24)
    ax = fig.add_subplot(grid[0, 0])
    table_ax = fig.add_subplot(grid[0, 1])
    plot_phase_curves(ax, reference, outputs, x0)
    add_timing_table(table_ax, timings)
    ax.set(xlabel=r"$q$", ylabel=r"$p$")
    ax.set_aspect("equal", adjustable="box")
    add_figure_legend(fig, ax)
    save_figure(fig, FIGURES / "harmonic_phase.png")
    return times, reference, outputs, timings


def main():
    configure_plots()
    _, reference_ho, outputs_ho, timings = plot_harmonic_phase()

    print("Harmonic oscillator: T=20, h=0.1")
    errors = {
        name: np.linalg.norm(states[-1] - reference_ho[-1])
        for name, states in outputs_ho.items()
    }
    print(
        "Final-state errors: "
        + ", ".join(f"{name}={error:.3e}" for name, error in errors.items())
    )
    print(
        "Integration timings: "
        + ", ".join(f"{name}={1000.0 * timings[name]:.2f} ms" for name in METHODS)
    )


if __name__ == "__main__":
    main()
