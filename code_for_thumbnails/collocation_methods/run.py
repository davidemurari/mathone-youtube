from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Fixing the configuration for the plots
plt.rcParams["figure.dpi"] = 300
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 13
plt.rcParams["figure.figsize"] = (6.4, 3.6)

# ------------------------------------------------------------
# Harmonic oscillator
#
# y' = A y,   y = (x,v)^T
# x' = v
# v' = -x
#
# y(0) = (0,1)^T
#
# Exact position: x(t) = sin(t)
# ------------------------------------------------------------

A = np.array([
    [0.0,  1.0],
    [-1.0, 0.0]
])

y0 = np.array([0.0, 1.0])

# Choose a reasonably large step so the difference is visible
h = 2.4

I = np.eye(2)

# ------------------------------------------------------------
# Implicit midpoint step
#
# y1 = y0 + h A ((y0+y1)/2)
#
# so
#
# (I - hA/2) y1 = (I + hA/2) y0
# ------------------------------------------------------------

y1 = np.linalg.solve(
    I - 0.5 * h * A,
    (I + 0.5 * h * A) @ y0
)

# ------------------------------------------------------------
# Exact solution
# ------------------------------------------------------------

def x_exact(t):
    return np.sin(t)

# ------------------------------------------------------------
# Collocation polynomial for implicit midpoint
#
# Since this is degree 1, it is just the line joining y0 and y1:
#
# p(t) = y0 + (t/h)(y1-y0),   0 <= t <= h
# ------------------------------------------------------------

def x_collocation(t):
    t = np.asarray(t)
    return y0[0] + (t / h) * (y1[0] - y0[0])

# Midpoint of the interval
t_mid = h / 2
x_mid = x_collocation(t_mid)

# Grids for plotting
t_exact_grid = np.linspace(0, 2*np.pi, 1000)
t_coll_grid = np.linspace(0, h, 300)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

cyan = "#00E5FF"
magenta = "#FF00CC"
yellow = "#FFE600"
white = "#FFFFFF"
gray = "#BFBFBF"
black = "#050505"

fig, ax = plt.subplots()

fig.patch.set_facecolor(black)
ax.set_facecolor(black)

# Exact solution
exact_line, = ax.plot(
    t_exact_grid,
    x_exact(t_exact_grid),
    color=cyan,
    lw=3.2,
    solid_capstyle="round",
    zorder=2,
)

# Implicit midpoint collocation polynomial
collocation_line, = ax.plot(
    t_coll_grid,
    x_collocation(t_coll_grid),
    color=magenta,
    lw=4.0,
    solid_capstyle="round",
    zorder=3,
)

# Horizontal axis
ax.axhline(0, color=white, lw=1.0, alpha=0.8, zorder=1)

# Vertical guides
ax.axvline(t_mid, color=gray, lw=1.0, ls=(0, (4, 4)), alpha=0.55, zorder=1)
ax.axvline(h, color=gray, lw=1.0, ls=(0, (4, 4)), alpha=0.55, zorder=1)

# Key points
ax.scatter(
    [0, t_mid, h],
    [y0[0], x_mid, y1[0]],
    color=yellow,
    edgecolor=black,
    linewidth=0.8,
    s=75,
    zorder=6,
)

# Put the curve labels together in the empty upper-right area.  A nearly opaque
# background keeps them readable even when the image is reduced to thumbnail size.
legend = ax.legend(
    [exact_line, collocation_line],
    [r"\textbf{exact solution}", r"\textbf{collocation polynomial}"],
    loc="upper right",
    bbox_to_anchor=(0.98, 0.97),
    fontsize=14,
    handlelength=2.4,
    handletextpad=0.7,
    labelspacing=0.6,
    borderpad=0.65,
    facecolor=black,
    edgecolor=gray,
    framealpha=0.94,
    fancybox=True,
)
legend.get_frame().set_linewidth(0.6)
legend.get_texts()[0].set_color(cyan)
legend.get_texts()[1].set_color(magenta)
legend.set_zorder(8)

ax.text(
    t_mid, -0.12,
    r"$h/2$",
    color=yellow,
    ha="center",
    va="top",
    fontsize=14,
    bbox=dict(facecolor=black, edgecolor="none", pad=1.0),
    zorder=7,
)

ax.text(
    h, -0.12,
    r"$h$",
    color=yellow,
    ha="center",
    va="top",
    fontsize=14,
    bbox=dict(facecolor=black, edgecolor="none", pad=1.0),
    zorder=7,
)

# Clean thumbnail styling
ax.set_xlim(-0.16, 2*np.pi + 0.12)
ax.set_ylim(-1.16, 1.16)

ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()

plt.savefig(
    Path(__file__).resolve().parent / "implicit_midpoint_thumbnail_plot.png",
    dpi=300,
    facecolor="black",
    bbox_inches="tight",
    pad_inches=0.02,
    transparent=False,
)

plt.show()
