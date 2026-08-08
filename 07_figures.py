from pathlib import Path
import infolattice as il
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter


ROOT = Path("/Users/matthiasflor/Documents/1. PhD/3. Information lattice/3d-info-latt")
STYLE_PATH = ROOT / "style.mplstyle"
FIGURES_DIR = ROOT / "figures"
N_SITES = (17, 17, 17)
REFERENCE_POINT = (8, 8, 8)
GRID_FACTOR = 6
TICKS = np.arange(0, 17, 4)
COLORBAR_LABEL = r"$i(\boldsymbol{n},\boldsymbol{n}')$"
COLORBAR_TICKS = 3
MARKER_SHRINK = 0.35
COLORBAR_WIDTH = 0.022
COLORBAR_HEIGHT = 0.34
COLORBAR_LEFT_SHIFT = .04
COLORBAR_BOTTOM = 0.35
COLORBAR_LABEL_X = -10
COLORBAR_LABEL_Y = .85
COLORBAR_LABEL_SCALE = 1.3
SELECTED_PLOT = "nodal"
SELECTED_PLOT = "dirac"
SELECTED_PLOT = "fermisurface"

PLOTS = {
    "fermisurface": {
        "run_dir": ROOT / "cluster" / "runs" / "tight_binding_17x17x17",
        "save_path": FIGURES_DIR / "i_fermisurface_17.png",
        "vmax": 0.0008,
        "cutoff": 0.3,
        "power": 1.1,
        "max_alpha": 0.9,
        "marker_scale": 80,
        "elev": 30,
        "azim": 160,
    },
    "nodal": {
        "run_dir": ROOT / "cluster" / "runs" / "nodal_line_17x17x17",
        "save_path": FIGURES_DIR / "i_nodal_17.png",
        "vmax": 0.0001,
        "cutoff": 0.3,
        "power": .7,
        "max_alpha": 0.9,
        "marker_scale": 80,
        "elev": 30,
        "azim": 160,
    },
    "dirac": {
        "run_dir": ROOT / "cluster" / "runs" / "pi_flux_17x17x17",
        "save_path": FIGURES_DIR / "i_dirac_17.png",
        "vmax": 0.0001,
        "cutoff": 0.1,
        "power": 0.7,
        "max_alpha": 0.9,
        "marker_scale": 80,
        "elev": 30,
        "azim": 160,
    },
}


def refine_linear_3d(arr, factor):
    arr = np.asarray(arr, dtype=float)
    x0 = np.arange(arr.shape[0], dtype=float)
    y0 = np.arange(arr.shape[1], dtype=float)
    z0 = np.arange(arr.shape[2], dtype=float)
    x1 = np.linspace(0.0, x0[-1], factor * (arr.shape[0] - 1) + 1)
    y1 = np.linspace(0.0, y0[-1], factor * (arr.shape[1] - 1) + 1)
    z1 = np.linspace(0.0, z0[-1], factor * (arr.shape[2] - 1) + 1)

    interp_x = np.empty((len(x1), arr.shape[1], arr.shape[2]), dtype=float)
    for j in range(arr.shape[1]):
        for k in range(arr.shape[2]):
            interp_x[:, j, k] = np.interp(x1, x0, arr[:, j, k])

    interp_xy = np.empty((len(x1), len(y1), arr.shape[2]), dtype=float)
    for i in range(len(x1)):
        for k in range(arr.shape[2]):
            interp_xy[i, :, k] = np.interp(y1, y0, interp_x[i, :, k])

    interp_xyz = np.empty((len(x1), len(y1), len(z1)), dtype=float)
    for i in range(len(x1)):
        for j in range(len(y1)):
            interp_xyz[i, j, :] = np.interp(z1, z0, interp_xy[i, j, :])

    return interp_xyz, x1, y1, z1


def build_colors(values, vmax, cutoff, power, max_alpha):
    vals = np.clip(values, -vmax, vmax)
    mag = np.abs(vals) / vmax
    alpha = max_alpha * mag ** power
    alpha[mag < cutoff] = 0.0
    keep = alpha > 0.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("bwr")
    colors = cmap(norm(vals[keep]))
    colors[:, 3] = alpha[keep]
    return keep, colors, norm, cmap, mag


def place_z_axis_left(ax):
    # mplot3d uses a private axis-layout switch to choose which edge carries the z axis.
    ax.zaxis._axinfo["juggled"] = (1, 2, 0)


def plot_interpolated_spread(field, x, y, z, config):
    fine_field, xf, yf, zf = refine_linear_3d(field, GRID_FACTOR)
    coords = np.meshgrid(xf, yf, zf, indexing="ij")
    values = fine_field.ravel()
    keep, colors, norm, cmap, mag = build_colors(
        values,
        config["vmax"],
        config["cutoff"],
        config["power"],
        config["max_alpha"],
    )
    size = max(0.3, MARKER_SHRINK * config["marker_scale"] / (GRID_FACTOR ** 2))
    order = np.argsort(mag[keep])

    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, COLORBAR_WIDTH), wspace=0.015)
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    cax = fig.add_subplot(gs[0, 1])

    ax.scatter(
        coords[0].ravel()[keep][order],
        coords[1].ravel()[keep][order],
        coords[2].ravel()[keep][order],
        c=colors[order],
        s=size,
        marker="o",
        depthshade=False,
        linewidths=0,
        rasterized=True,
    )

    ax.set_xlim(-0.5, N_SITES[0] - 0.5)
    ax.set_ylim(-0.5, N_SITES[1] - 0.5)
    ax.set_zlim(-0.5, N_SITES[2] - 0.5)
    ax.set_box_aspect(N_SITES)
    ax.view_init(elev=config["elev"], azim=config["azim"])
    ax.set_xlabel(r"$n_x'$", labelpad=12)
    ax.set_ylabel(r"$n_y'$", labelpad=12)
    ax.set_zlabel(r"$n_z'$", labelpad=14)
    ax.set_xticks(TICKS)
    ax.set_yticks(TICKS)
    ax.set_zticks(TICKS)
    place_z_axis_left(ax)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, cax=cax)
    box = cax.get_position()
    cax.set_position([
        box.x0 - COLORBAR_LEFT_SHIFT,
        COLORBAR_BOTTOM,
        box.width,
        COLORBAR_HEIGHT,
    ])
    tick_values = np.linspace(-config["vmax"], config["vmax"], COLORBAR_TICKS)
    cbar.set_ticks(tick_values)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    cbar.formatter = formatter
    cbar.update_ticks()
    cbar.set_label("")
    label_x = box.x0 + COLORBAR_LABEL_X * box.width
    label_y = box.y0 + COLORBAR_LABEL_Y * box.height
    fig.text(
        label_x,
        label_y,
        COLORBAR_LABEL,
        transform=fig.transFigure,
        ha="left",
        va="bottom",
        fontsize=COLORBAR_LABEL_SCALE * plt.rcParams["axes.labelsize"],
    )
    cbar.ax.yaxis.get_offset_text().set_size(plt.rcParams["axes.labelsize"])

    config["save_path"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config["save_path"], dpi=300, bbox_inches="tight")
    plt.show()
plt.style.use(STYLE_PATH)

config = PLOTS[SELECTED_PLOT]
manifest = config["run_dir"] / "manifest.json"
lat = il.InformationLattice(N_SITES, precompute_subsystems=False)
lat.load_slurm_results(manifest)
local_information_spread = lat.spread_around_point(*REFERENCE_POINT)
plot_interpolated_spread(local_information_spread, *REFERENCE_POINT, config)
print(f"Saved {config['save_path']}")
