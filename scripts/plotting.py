"""The common compact Mathone plotting style."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import Collection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text


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
YELLOW = "#FFF275"

WHITE_REFERENCE_COLOR = "#111111"
WHITE_TEXT = "#111111"
WHITE_MUTED_TEXT = "#333333"
WHITE_GRID = "#D0D7DE"
WHITE_GUIDE_COLOR = "#666666"
WHITE_CYAN = "#0072B2"
WHITE_RED = "#D55E00"
WHITE_GREEN = "#009E73"
WHITE_PINK = "#CC79A7"
WHITE_PURPLE = "#6A3D9A"
WHITE_YELLOW = "#C49A00"

WHITE_COLOR_MAP = {
    REFERENCE_COLOR: WHITE_REFERENCE_COLOR,
    LIGHT_TEXT: WHITE_TEXT,
    MUTED_TEXT: WHITE_MUTED_TEXT,
    LIGHT_GRID: WHITE_GRID,
    GUIDE_COLOR: WHITE_GUIDE_COLOR,
    CYAN: WHITE_CYAN,
    RED: WHITE_RED,
    GREEN: WHITE_GREEN,
    PINK: WHITE_PINK,
    PURPLE: WHITE_PURPLE,
    YELLOW: WHITE_YELLOW,
}


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
        color=[CYAN, RED, GREEN, PINK, PURPLE, YELLOW]
    )
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["lines.linewidth"] = 1.55
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.labelcolor"] = LIGHT_TEXT
    plt.rcParams["xtick.color"] = MUTED_TEXT
    plt.rcParams["ytick.color"] = MUTED_TEXT
    plt.rcParams["text.color"] = LIGHT_TEXT
    plt.rcParams["grid.color"] = LIGHT_GRID


def _map_color(color):
    try:
        rgba = mcolors.to_rgba(color)
    except (TypeError, ValueError):
        return color

    for old, new in WHITE_COLOR_MAP.items():
        if all(abs(a - b) < 1e-8 for a, b in zip(rgba, mcolors.to_rgba(old))):
            return new
    return color


def _map_color_array(colors):
    if len(colors) == 0:
        return colors
    return [_map_color(color) for color in colors]


def _apply_white_style(fig):
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)

    for ax in fig.axes:
        ax.patch.set_facecolor("white")
        ax.patch.set_alpha(1.0)
        ax.tick_params(colors=WHITE_MUTED_TEXT)
        ax.xaxis.label.set_color(WHITE_TEXT)
        ax.yaxis.label.set_color(WHITE_TEXT)
        ax.title.set_color(WHITE_TEXT)
        for spine in ax.spines.values():
            spine.set_color(WHITE_TEXT)

    for line in fig.findobj(Line2D):
        line.set_color(_map_color(line.get_color()))

    for text in fig.findobj(Text):
        text.set_color(_map_color(text.get_color()))

    for collection in fig.findobj(Collection):
        collection.set_facecolor(_map_color_array(collection.get_facecolors()))
        collection.set_edgecolor(_map_color_array(collection.get_edgecolors()))

    for patch in fig.findobj(Patch):
        patch.set_edgecolor(_map_color(patch.get_edgecolor()))
        patch.set_facecolor(_map_color(patch.get_facecolor()))


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.patch.set_alpha(0.0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03, transparent=True)

    white_path = path.parent / "white" / path.name
    white_path.parent.mkdir(parents=True, exist_ok=True)
    _apply_white_style(fig)
    fig.savefig(
        white_path,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(fig)
