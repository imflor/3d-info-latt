from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from matplotlib.ticker import NullLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


ROOT = Path("/Users/matthiasflor/Documents/1. PhD/3. Information lattice/3d-info-latt")
STYLE_PATH = ROOT / "style.mplstyle"
FIGURES_DIR = ROOT / "figures"
SAVE_PATH = FIGURES_DIR / "kspace_fermisurface.png"

VIEW_ELEV = 30
VIEW_AZIM = 160
FIGSIZE = (7.0, 6.0)
SURFACE_POINTS = 96
SURFACE_COLOR = "0.85"
LIGHT_SOURCE = LightSource(azdeg=315, altdeg=35)
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

k = np.linspace(-np.pi, np.pi, SURFACE_POINTS)
KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
energy = np.cos(KX) + np.cos(KY) + np.cos(KZ)
spacing = (k[1] - k[0],) * 3

verts, faces, _, _ = marching_cubes(
    energy.astype(np.float32),
    level=0.0,
    spacing=spacing,
)
verts += np.array([k[0], k[0], k[0]])

face_verts = verts[faces]
mesh = Poly3DCollection(
    face_verts,
    facecolors=[SURFACE_COLOR],
    edgecolors=[SURFACE_COLOR],
    linewidths=0.0,
    shade=True,
    lightsource=LIGHT_SOURCE,
)
mesh.set_edgecolor("none")

fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot(projection="3d")
ax.add_collection3d(mesh)

style_axes(ax)
fig.savefig(SAVE_PATH, dpi=300)
plt.show()
print(f"Saved {SAVE_PATH}")
