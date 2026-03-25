import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm


def _colored_points(arr, vmax, cutoff, power, max_alpha):
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
    vals = np.clip(arr.ravel(), -vmax, vmax)
    mag = np.abs(vals) / vmax
    alpha = max_alpha * mag ** power
    alpha[mag < cutoff] = 0.0
    keep = alpha > 0

    colors = cmap(norm(vals[keep]))
    colors[:, 3] = alpha[keep]
    return (
        x.ravel()[keep],
        y.ravel()[keep],
        z.ravel()[keep],
        colors,
        arr.shape,
    )


def plot_3d_array(arr, vmax=0.7, cutoff=0.01, power=0.5, max_alpha=0.95,
                  marker_scale=140, view=(10, -70), marker="s", show=True):
    x, y, z, colors, shape = _colored_points(arr, vmax, cutoff, power, max_alpha)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        x,
        y,
        z,
        c=colors,
        s=marker_scale,
        marker=marker,
        depthshade=False,
        linewidths=0,
    )

    ax.set_xlim(-0.5, shape[0] - 0.5)
    ax.set_ylim(-0.5, shape[1] - 0.5)
    ax.set_zlim(-0.5, shape[2] - 0.5)
    ax.set_box_aspect(shape)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    if show and "agg" not in plt.get_backend().lower():
        plt.show()
    return fig, ax


def save_rotating_3d_array(
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
    x, y, z, colors, shape = _colored_points(arr, vmax, cutoff, power, max_alpha)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        x,
        y,
        z,
        c=colors,
        s=marker_scale,
        marker=marker,
        depthshade=False,
        linewidths=0,
    )

    ax.set_xlim(-0.5, shape[0] - 0.5)
    ax.set_ylim(-0.5, shape[1] - 0.5)
    ax.set_zlim(-0.5, shape[2] - 0.5)
    ax.set_box_aspect(shape)
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
    return save_path
