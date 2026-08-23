"""The common compact Mathone plotting style."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LIGHT_TEXT = "#F3F7FA"
MUTED_TEXT = "#C8D1D9"
LIGHT_GRID = "#54616C"
REFERENCE_COLOR = "#F2F7FF"
GUIDE_COLOR = "#AEB8C2"
CYAN = "#00D7FF"
RED = "#FF453A"
GREEN = "#30F29A"
PINK = "#FF4FD8"
PURPLE = "#B967FF"


def configure_plots():
    # Fixing the configuration for the plots
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 10
    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["figure.edgecolor"] = "none"
    plt.rcParams["savefig.facecolor"] = "none"
    plt.rcParams["savefig.edgecolor"] = "none"
    plt.rcParams["axes.facecolor"] = "none"
    plt.rcParams["axes.edgecolor"] = LIGHT_TEXT
    plt.rcParams["axes.labelcolor"] = LIGHT_TEXT
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=[CYAN, RED, GREEN, PINK, PURPLE, "#FFF275"]
    )
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["lines.linewidth"] = 1.55
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.labelcolor"] = LIGHT_TEXT
    plt.rcParams["xtick.color"] = MUTED_TEXT
    plt.rcParams["ytick.color"] = MUTED_TEXT
    plt.rcParams["text.color"] = LIGHT_TEXT
    plt.rcParams["grid.color"] = LIGHT_GRID


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.patch.set_alpha(0.0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03, transparent=True)
    plt.close(fig)
