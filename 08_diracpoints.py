from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator


ROOT = Path("/Users/matthiasflor/Documents/1. PhD/3. Information lattice/3d-info-latt")
STYLE_PATH = ROOT / "style.mplstyle"
FIGURES_DIR = ROOT / "figures"
SAVE_PATH = FIGURES_DIR / "kspace_diracpoints.png"

VIEW_ELEV = 30
VIEW_AZIM = 160
FIGSIZE = (7.0, 6.0)
MARKER_COLOR = "0.20"
MARKER_SIZE = 42
TICKS = [-np.pi, 0.0, np.pi]
TICK_LABELS = [r"$-\pi$", r"$0$", r"$\pi$"]


def place_z_axis_left(ax):
    ax.zaxis._axinfo["juggled"] = (1, 2, 0)


def style_axes(ax):
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_zlim(-np.pi, np.pi)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_xlabel(r"$k_x$", labelpad=12)
    ax.set_ylabel(r"$k_y$", labelpad=12)
    ax.set_zlabel(r"$k_z$", labelpad=14)
    ax.set_xticks(TICKS, TICK_LABELS)
    ax.set_yticks(TICKS, TICK_LABELS)
    ax.set_zticks(TICKS, TICK_LABELS)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.zaxis.set_minor_locator(NullLocator())
    place_z_axis_left(ax)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))


plt.style.use(STYLE_PATH)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

points = np.array(list(product([-np.pi / 2, np.pi / 2], repeat=3)), dtype=float)

fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot(projection="3d")
ax.scatter(
    points[:, 0],
    points[:, 1],
    points[:, 2],
    s=MARKER_SIZE,
    color=MARKER_COLOR,
    depthshade=False,
)

style_axes(ax)
fig.savefig(SAVE_PATH, dpi=300)
plt.show()
print(f"Saved {SAVE_PATH}")
