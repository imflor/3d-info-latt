from pathlib import Path
import sys

if "__file__" in globals():
    here = Path(__file__).resolve().parent
else:
    here = Path.cwd()
    if not (here / "physics.py").exists():
        if (here / "demo" / "physics.py").exists():
            here = here / "demo"

if str(here) not in sys.path:
    sys.path.insert(0, str(here))

import physics
import lattice

ENABLE_PARALLEL = True
ENABLE_LOADER = True

# Initialize state and wave function
n_qubits = [8, 8, 8]

# State
state = physics.TightBindingGS(n_qubits, t=1)

# Information lattice
lat = lattice.InformationLattice(n_qubits, parallel=ENABLE_PARALLEL, loader=ENABLE_LOADER)
lat.compute(state)

# Multiscale information
i = lat.i_local.sum(axis=(3, 4, 5))

##

import matplotlib.pyplot as plt


def plot_3d_matplotlib(arr, vmax=0.7, cutoff=.01, power=.5, max_alpha=0.95,
                       marker_scale=140, view=(10, -70), marker="s"):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 3:
        raise ValueError("arr must be a 3D array")

    if vmax is None:
        vmax = np.max(np.abs(arr))
    if vmax <= 0:
        vmax = 1.0

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("bwr")

    x, y, z = np.indices(arr.shape)
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    vals = np.clip(arr.ravel(), -vmax, vmax)

    mag = np.abs(vals) / vmax
    alpha = max_alpha * mag**power
    alpha[mag < cutoff] = 0.0

    keep = alpha > 0
    colors = cmap(norm(vals[keep]))
    colors[:, 3] = alpha[keep]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        x[keep], y[keep], z[keep],
        c=colors,
        s=marker_scale,
        marker=marker,
        depthshade=False,
        linewidths=0
    )

    ax.set_xlim(-0.5, arr.shape[0] - 0.5)
    ax.set_ylim(-0.5, arr.shape[1] - 0.5)
    ax.set_zlim(-0.5, arr.shape[2] - 0.5)
    ax.set_box_aspect(arr.shape)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.show()

plot_3d_matplotlib(i)

##

vmax = 10

plt.figure(figsize=(5, 5))
plt.imshow(i.sum(axis=0).T, cmap="bwr", vmin=-vmax, vmax=vmax, origin="lower")
plt.xlabel(r"\ell_y")
plt.xlabel(r"\ell_z")
plt.show()

##

i_corner = lat.i_local[:, :, :, 0, 0, 0]
plot_3d_matplotlib(i_corner, vmax=.006, cutoff=.1, power=.5, max_alpha=.5,
                   marker_scale=300, view=(30, 170), marker="s")


##

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FuncAnimation, PillowWriter

def rotate_3d_matplotlib_video(
    arr,
    save_path="figures/rotation.gif",
    vmax=None,
    cutoff=0.02,
    power=0.8,
    max_alpha=0.95,
    marker_scale=140,
    elev=30,
    azim_start=170,
    azim_stop=530,
    frames=120,
    fps=20,
    marker="s",
):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 3:
        raise ValueError("arr must be a 3D array")

    if vmax is None:
        vmax = np.max(np.abs(arr))
    if vmax <= 0:
        vmax = 1.0

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("bwr")

    x, y, z = np.indices(arr.shape)
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    vals = np.clip(arr.ravel(), -vmax, vmax)

    mag = np.abs(vals) / vmax
    alpha = max_alpha * mag**power
    alpha[mag < cutoff] = 0.0

    keep = alpha > 0
    colors = cmap(norm(vals[keep]))
    colors[:, 3] = alpha[keep]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        x[keep], y[keep], z[keep],
        c=colors,
        s=marker_scale,
        marker=marker,
        depthshade=False,
        linewidths=0
    )

    ax.set_xlim(-0.5, arr.shape[0] - 0.5)
    ax.set_ylim(-0.5, arr.shape[1] - 0.5)
    ax.set_zlim(-0.5, arr.shape[2] - 0.5)
    ax.set_box_aspect(arr.shape)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    azims = np.linspace(azim_start, azim_stop, frames)

    def update(i):
        ax.view_init(elev=elev, azim=azims[i])
        return fig,

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

i_corner = lat.i_local[:, :, :, 0, 0, 0]

rotate_3d_matplotlib_video(
    i_corner,
    save_path="figures/i_corner_rotation.gif",
    vmax=0.006,
    cutoff=0.1,
    power=0.5,
    max_alpha=0.5,
    marker_scale=300,
    elev=30,
    azim_start=170,
    azim_stop=530,
    frames=120,
    fps=20,
    marker="s",
)

##

from scipy.optimize import curve_fit

y = np.array([lat.i_local[i, i, i, 0, 0, 0] for i in range(min(n_qubits))])
x = np.arange(y.size)
m = np.where(x > 3)[0]
pos = np.where(y > 0)[0]
index = np.intersect1d(m, pos)

(a,), _ = curve_fit(lambda x, a: a * x**-2, x[index], y[index])

plt.figure(figsize=(3, 3))
plt.loglog(x, a * x**-2., "k-", label=r'$\propto \ell^{-2}$', lw=1)
plt.loglog(x[pos], y[pos], ".-", c=(1,0,1), lw=1, markersize=8)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$i_{(0\ 0)}^{(\ell_x\ \ell_y)}$")
plt.legend()
plt.ylim([1e-3, .2])
plt.savefig("figures/i_corner_fit.png", bbox_inches="tight", dpi=300)
plt.show()

##

import numpy as np


def i_loc_at_corner(i_loc, corner, a, b, c, fill=np.nan, flip=True, crop=True):
    """
    Extract the 3D local-information block with site (a,b,c) fixed at a chosen corner.

    Corner labels use the order (y, x, z):
        blf, brf, tlf, trf, blb, brb, tlb, trb
    """
    Nx, Ny, Nz = i_loc.shape[:3]
    Lx, Ly, Lz = np.meshgrid(
        np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij"
    )

    cy, cx, cz = corner[0], corner[1], corner[2]

    A = a - (cx == "r") * Lx
    B = b - (cy == "t") * Ly
    C = c - (cz == "b") * Lz

    valid = (
        (A >= 0) & (B >= 0) & (C >= 0)
        & (A < (Nx - Lx))
        & (B < (Ny - Ly))
        & (C < (Nz - Lz))
    )

    if crop:
        ix = np.where(valid.any(axis=(1, 2)))[0]
        iy = np.where(valid.any(axis=(0, 2)))[0]
        iz = np.where(valid.any(axis=(0, 1)))[0]
        I = np.ix_(ix, iy, iz)
        out = i_loc[Lx[I], Ly[I], Lz[I], A[I], B[I], C[I]]
    else:
        out = np.full((Nx, Ny, Nz), fill, dtype=float)
        out[valid] = i_loc[Lx[valid], Ly[valid], Lz[valid], A[valid], B[valid], C[valid]]

    M = out
    if flip:
        if cx == "r":
            M = M[::-1, :, :]
        if cy == "t":
            M = M[:, ::-1, :]
        if cz == "b":
            M = M[:, :, ::-1]
    return M


def stitch_all_directions(i_loc, a, b, c, fill=np.nan):
    """
    Stitch the 8 corner-based 3D views into one 3D array.

    The result is the 3D analogue of the 2D stitched map: the reference point
    (a,b,c) becomes the central meeting point of the 8 octants.
    """
    blocks = {
        "blf": i_loc_at_corner(i_loc, "blf", a, b, c, fill=fill, flip=True, crop=True),
        "brf": i_loc_at_corner(i_loc, "brf", a, b, c, fill=fill, flip=True, crop=True),
        "tlf": i_loc_at_corner(i_loc, "tlf", a, b, c, fill=fill, flip=True, crop=True),
        "trf": i_loc_at_corner(i_loc, "trf", a, b, c, fill=fill, flip=True, crop=True),
        "blb": i_loc_at_corner(i_loc, "blb", a, b, c, fill=fill, flip=True, crop=True),
        "brb": i_loc_at_corner(i_loc, "brb", a, b, c, fill=fill, flip=True, crop=True),
        "tlb": i_loc_at_corner(i_loc, "tlb", a, b, c, fill=fill, flip=True, crop=True),
        "trb": i_loc_at_corner(i_loc, "trb", a, b, c, fill=fill, flip=True, crop=True),
    }

    out = np.full(i_loc.shape[:3], fill, dtype=float)

    cx, cy, cz = blocks["trb"].shape

    for corner, M in blocks.items():
        sy, sx, sz = corner[0], corner[1], corner[2]

        xs = slice(None, cx) if sx == "r" else slice(cx - 1, None)
        ys = slice(None, cy) if sy == "t" else slice(cy - 1, None)
        zs = slice(None, cz) if sz == "b" else slice(cz - 1, None)

        out[xs, ys, zs] = M

    return out


rotate_3d_matplotlib_video(
    stitch_all_directions(lat.i_local, 5, 5, 5),
    save_path="figures/i_middle.gif",
    vmax=0.004,
    cutoff=0.01,
    power=2,
    max_alpha=1,
    marker_scale=200,
    elev=30,
    azim_start=170,
    azim_stop=530,
    frames=120,
    fps=20,
    marker="o",
)

plot_3d_matplotlib(stitch_all_directions(lat.i_local, 5, 5, 5),
                   vmax=.006, cutoff=.1, power=.5, max_alpha=.5,
                   marker_scale=300, view=(30, 170), marker="s")
