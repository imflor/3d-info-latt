import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def discrete_cmap_norm(values, colors):
    v = np.unique(np.round(values, decimals=5))
    if len(colors) < v.size:
        raise ValueError(f"Need {v.size} colors for values {v}, got only {len(colors)}.")
    if v.size == 1:
        b = np.array([v[0] - 0.5, v[0] + 0.5])
    else:
        b = np.r_[v[0] - (v[1] - v[0]) / 2, (v[:-1] + v[1:]) / 2, v[-1] + (v[-1] - v[-2]) / 2]
    cmap = ListedColormap(colors[:len(v)])
    return v, b, cmap, BoundaryNorm(b, cmap.N)


def plot_infolattice(i_local, colors, pad_num=0.04, pad_lab=0.08, cbar_rect=(0.35, 0.93, 0.30, 0.02)):
    """
    Plot the information lattice for a finite system on a 2D square lattice.

    Parameters
    ----------
    i_local : np.ndarray
        Array indexed as i_local[lx, ly, nx, ny].
    pad_num : float
        Figure-coordinate padding used to place the numeric (lx, ly) tick labels outside the grid.
    pad_lab : float
        Figure-coordinate padding used to place the axis labels (ell_x, ell_y) outside the numeric labels.
    cbar_rect : tuple[float, float, float, float]
        [left, bottom, width, height] for the colorbar axes in figure coordinates.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_map : dict[tuple[int,int], matplotlib.axes.Axes]
        Mapping (lx, ly) -> axes.
    """

    Nx = i_local.shape[0]
    Ny = i_local.shape[1]

    # colormap settings
    values, bounds, cmap, norm = discrete_cmap_norm(i_local, colors)

    # figure
    fig = plt.figure(figsize=(4, 4), constrained_layout=False)
    gs = fig.add_gridspec(
        Ny, Nx,
        width_ratios=np.arange(Nx, 0, -1),
        height_ratios=np.arange(1, Ny + 1),
        hspace=0.2, wspace=0.2,
    )

    ax_map = {}
    for lx in range(Nx):
        for ly in range(Ny):
            ax = fig.add_subplot(gs[Ny - ly - 1, lx])
            ax_map[(lx, ly)] = ax
            ax.set(xticks=[], yticks=[])
            for s in ax.spines.values():
                s.set(color="k", linewidth=0.75)
            if lx == 0 and ly == 0:
                ax.text(1.02, -0.06, r"$n_x$", transform=ax.transAxes, ha="right", va="top")
                ax.text(-0.06, 1.02, r"$n_y$", transform=ax.transAxes, ha="right", va="top", rotation=90)

            max_nx = Nx - lx
            max_ny = Ny - ly
            if ly == 0:
                ax.set_xticks(2 * np.arange(max_nx))
                ax.set_xticklabels([])
            if lx == 0:
                ax.set_yticks(2 * np.arange(max_ny))
                ax.set_yticklabels([])

            i_loc = i_local[lx, ly, :max_nx, :max_ny]
            x_edges = 2 * np.arange(max_nx + 1) - 1
            y_edges = 2 * np.arange(max_ny + 1) - 1
            ax.pcolormesh(
                x_edges, y_edges, i_loc.T,
                cmap=cmap, norm=norm,
                edgecolors="0.85", linewidth=0.35,
                shading="flat", zorder=100,
            )
            ax.set(xlim=(x_edges[0], x_edges[-1]), ylim=(y_edges[0], y_edges[-1]))

    # external (lx, ly) labels
    fig.canvas.draw()
    for lx in range(Nx):
        bb = ax_map[(lx, 0)].get_position()
        fig.text(0.5 * (bb.x0 + bb.x1), bb.y0 - pad_num, rf"${lx}$", ha="center", va="top")
    for ly in range(Ny):
        bb = ax_map[(0, ly)].get_position()
        fig.text(bb.x0 - pad_num, 0.5 * (bb.y0 + bb.y1), rf"${ly}$", ha="right", va="center")
    bb_left = ax_map[(0, 0)].get_position()
    bb_right = ax_map[(Nx-1, 0)].get_position()
    fig.text(0.5 * (bb_left.x0 + bb_right.x1), bb_left.y0 - pad_lab, r"$\ell_x$", ha="center", va="top")
    bb_bot = ax_map[(0, 0)].get_position()
    bb_top = ax_map[(0, Ny-1)].get_position()
    fig.text(
        bb_bot.x0 - pad_lab,
        0.5 * (bb_bot.y0 + bb_top.y1),
        r"$\ell_y$",
        rotation=90,
        ha="center",
        va="center",
    )

    # colorbar
    fig.subplots_adjust(top=0.90)
    centers = 0.5 * (bounds[:-1] + bounds[1:])
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes(list(cbar_rect))
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", drawedges=True)
    cbar.set_ticks(centers)
    cbar.set_ticklabels([f"{v:g}" for v in values])
    cbar.ax.xaxis.set_ticks_position("top")

    return fig, ax_map


def plot_infolattice_toric_code(i_local, colors, pad_num=0.04, pad_lab=0.08, cbar_rect=(0.35, 0.93, 0.30, 0.02)):
    """
    Plot the information lattice of the toric code.

    Parameters
    ----------
    i_local : np.ndarray
        Array indexed as i_local[lx, ly, nx, ny].
    pad_num : float
        Figure-coordinate padding used to place the numeric (lx, ly) tick labels outside the grid.
    pad_lab : float
        Figure-coordinate padding used to place the axis labels (ell_x, ell_y) outside the numeric labels.
    cbar_rect : tuple[float, float, float, float]
        [left, bottom, width, height] for the colorbar axes in figure coordinates.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_map : dict[tuple[int,int], matplotlib.axes.Axes]
        Mapping (lx, ly) -> axes.
    """

    Nx = i_local.shape[0] - 1
    Ny = i_local.shape[1] - 1

    # colormap settings
    values, bounds, cmap, norm = discrete_cmap_norm(i_local, colors)

    # figure
    fig = plt.figure(figsize=(4, 4), constrained_layout=False)
    gs = fig.add_gridspec(
        Ny + 1, Nx + 1,
        width_ratios=np.arange(Nx + 1, 0, -1),
        height_ratios=np.arange(1, Ny + 2),
        hspace=0.2, wspace=0.2,
    )

    ax_map = {}
    for lx in range(Nx + 1):
        for ly in range(Ny + 1):
            ax = fig.add_subplot(gs[Ny - ly, lx])
            ax_map[(lx, ly)] = ax
            ax.set(xticks=[], yticks=[])
            for s in ax.spines.values():
                s.set(color="k", linewidth=0.75)
            if lx == 0 and ly == 0:
                ax.text(1.02, -0.06, r"$n_x$", transform=ax.transAxes, ha="right", va="top")
                ax.text(-0.06, 1.02, r"$n_y$", transform=ax.transAxes, ha="right", va="top", rotation=90)
                ax.spines[["right", "top"]].set_visible(False)
                continue

            max_nx = Nx - lx + 1
            max_ny = Ny - ly + 1
            if ly == 0:
                ax.set_xticks(2 * np.arange(max_nx))
                ax.set_xticklabels([])
            if lx == 0:
                ax.set_yticks(2 * np.arange(max_ny))
                ax.set_yticklabels([])

            i_loc = i_local[lx, ly, :max_nx, :max_ny]
            x_edges = 2 * np.arange(max_nx + 1) - 1
            y_edges = 2 * np.arange(max_ny + 1) - 1
            ax.pcolormesh(
                x_edges, y_edges, i_loc.T,
                cmap=cmap, norm=norm,
                edgecolors="0.85", linewidth=0.35,
                shading="flat", zorder=100,
            )
            ax.set(xlim=(x_edges[0], x_edges[-1]), ylim=(y_edges[0], y_edges[-1]))

    # external (lx, ly) labels
    fig.canvas.draw()
    for lx in range(Nx + 1):
        bb = ax_map[(lx, 0)].get_position()
        fig.text(0.5 * (bb.x0 + bb.x1), bb.y0 - pad_num, rf"${lx}$", ha="center", va="top")
    for ly in range(Ny + 1):
        bb = ax_map[(0, ly)].get_position()
        fig.text(bb.x0 - pad_num, 0.5 * (bb.y0 + bb.y1), rf"${ly}$", ha="right", va="center")
    bb_left = ax_map[(0, 0)].get_position()
    bb_right = ax_map[(Nx, 0)].get_position()
    fig.text(0.5 * (bb_left.x0 + bb_right.x1), bb_left.y0 - pad_lab, r"$\ell_x$", ha="center", va="top")
    bb_bot = ax_map[(0, 0)].get_position()
    bb_top = ax_map[(0, Ny)].get_position()
    fig.text(
        bb_bot.x0 - pad_lab,
        0.5 * (bb_bot.y0 + bb_top.y1),
        r"$\ell_y$",
        rotation=90,
        ha="center",
        va="center",
    )

    # colorbar
    fig.subplots_adjust(top=0.90)
    centers = 0.5 * (bounds[:-1] + bounds[1:])
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes(list(cbar_rect))
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", drawedges=True)
    cbar.set_ticks(centers)
    cbar.set_ticklabels([f"{v:g}" for v in values])
    cbar.ax.xaxis.set_ticks_position("top")

    return fig, ax_map
