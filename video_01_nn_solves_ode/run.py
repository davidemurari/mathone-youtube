"""Train one global neural solution on short and long time intervals."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from scripts.ode_systems import damped_oscillator
from scripts.pinn import evaluate_pinn, train_damped_oscillator_pinn
from scripts.plotting import CYAN, REFERENCE_COLOR, configure_plots, save_figure
from scripts.reference_solver import reference_solution


GAMMA = 0.05
X0 = np.array([1.0, 0.0])
HORIZONS = (10.0, 40.0)
EPOCHS = 4000
LBFGS_STEPS = 400
POINTS_PER_UNIT = 16
FIGURES = Path(__file__).resolve().parent / "figures"


def plot_position(times, reference, prediction, horizon, path):
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    ax.plot(times, reference[:, 0], color=REFERENCE_COLOR, label="reference")
    ax.plot(times, prediction[:, 0], color=CYAN, linestyle="--", label="NN")
    ax.set(xlabel=r"$t$", ylabel=r"$q(t)$", xlim=(0.0, horizon))
    ax.legend(
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0.0,
        fontsize=8,
    )
    save_figure(fig, path)


def plot_phase(reference, prediction, path):
    fig, ax = plt.subplots(figsize=(2.75, 2.55))
    ax.plot(reference[:, 0], reference[:, 1], color=REFERENCE_COLOR, label="reference")
    ax.plot(prediction[:, 0], prediction[:, 1], color=CYAN, linestyle="--", label="NN")
    ax.scatter([X0[0]], [X0[1]], s=12, color=REFERENCE_COLOR, zorder=3)
    ax.set(xlabel=r"$q$", ylabel=r"$p$")
    ax.legend(
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0.0,
        fontsize=8,
    )
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, path)


def run_horizon(horizon, label):
    result = train_damped_oscillator_pinn(
        horizon,
        gamma=GAMMA,
        x0=X0,
        epochs=EPOCHS,
        points_per_unit=POINTS_PER_UNIT,
        seed=7,
        lbfgs_steps=LBFGS_STEPS,
    )
    # This deterministic dense grid is independent of the random training points
    # and is used for both reference comparison and plots.
    times = np.linspace(0.0, horizon, int(200 * horizon) + 2)
    reference = reference_solution(damped_oscillator, X0, times, args=(GAMMA,))
    prediction = evaluate_pinn(result, times)
    pointwise_error = np.linalg.norm(prediction - reference, axis=1)
    rms_error = np.sqrt(np.mean(pointwise_error**2))
    max_error = np.max(pointwise_error)

    plot_position(times, reference, prediction, horizon, FIGURES / f"{label}_position.png")
    plot_phase(reference, prediction, FIGURES / f"{label}_phase.png")
    print(
        f"T={horizon:4.1f} | residual loss={result.final_loss:.3e} | "
        f"RMS state error={rms_error:.3e} | max state error={max_error:.3e}"
    )
    return rms_error, max_error


def main():
    configure_plots()
    print(
        f"PINN: width=32, depth=3, epochs={EPOCHS}, "
        f"L-BFGS steps={LBFGS_STEPS}, "
        f"collocation density={POINTS_PER_UNIT}/unit, seed=7"
    )
    short = run_horizon(HORIZONS[0], "short")
    long = run_horizon(HORIZONS[1], "long")
    print(f"Long/short RMS error ratio: {long[0] / short[0]:.2f}")


if __name__ == "__main__":
    main()
