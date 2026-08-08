from pathlib import Path
import infolattice as il
import matplotlib.pyplot as plt

run_dir = Path("/Users/matthiasflor/Documents/1. PhD/3. Information lattice/3d-info-latt/cluster/runs/tight_binding_15x15x15")
manifest = run_dir / "manifest.json"

n_sites = (15, 15, 15)
lat = il.InformationLattice(n_sites)
lat.load_slurm_results(manifest)

## Plot the information per multi-scale

information_per_scale = lat.i_local.sum(axis=(3, 4, 5))
fig, ax = il.plot_3d_array(
    information_per_scale,
    vmax=1,
    cutoff=.001,
    power=2,
    max_alpha=1,
    marker_scale=180,
    view=(20, -55),
    marker="s",
)
plt.show()


## Plot the local information spread

reference_point = (7, 7, 7)
local_information_spread = lat.spread_around_point(*reference_point)
il.save_rotating_3d_array(
    local_information_spread,
    save_path="figures/i_centered_15.gif",
    vmax=.002,
    cutoff=.1,
    power=.9,
    max_alpha=1,
    marker_scale=100,
    marker="o",
    elev=30,
    azim_start=170,
    azim_stop=530,
    frames=120,
    fps=20,
)

##

# NOTE
# I AM NOT AT ALL SURE THIS SUMMATION MAKES ANY SENSE WHATSOEVER!!!
# NOTE THAT IT'S NOT THE LOCAL INFORMATION

fig, ax = plt.subplots(1, 3, figsize=(9, 3))
fig.suptitle(r"$i(\boldsymbol{n}, \boldsymbol{n}')$ with $\boldsymbol{n}'=(7,0,7)$")
ax[0].imshow(local_information_spread.sum(axis=2).T, cmap="bwr", vmin=-.1, vmax=.1, origin="lower")
ax[0].set_title(r'Projection on the $xy$-plane')
ax[0].set_xlabel(r'$n_x$')
ax[0].set_ylabel(r'$n_y$')
ax[1].imshow(local_information_spread.sum(axis=1).T, cmap="bwr", vmin=-.1, vmax=.1, origin="lower")
ax[1].set_title(r'Projection on the $xz$-plane')
ax[1].set_xlabel(r'$n_x$')
ax[1].set_ylabel(r'$n_z$')
ax[2].imshow(local_information_spread.sum(axis=0).T, cmap="bwr", vmin=-.1, vmax=.1, origin="lower")
ax[2].set_title(r'Projection on the $yz$-plane')
ax[2].set_xlabel(r'$n_y$')
ax[2].set_ylabel(r'$n_z$')
plt.tight_layout()
plt.savefig("figures/i_projections_15.png", dpi=300)
plt.show()

##

# IMPORTANT NOTE:
# MY FIRST THOUGHT WAS THAT THERE ARE NO REFLECTIONS
# BUT ACTUALLY, THEY ARE OUT OF THE PLANE SO WE DON'T SEE THEM HERE
# THE BEST WOULD BE ACTUALLY TO CONSIDER SLICES THAT ARE ORIENTED AT 45 DEG

# --> FUTURE TODO

fig, ax = plt.subplots(1, 3, figsize=(9, 3))
fig.suptitle(r"$i(\boldsymbol{n}, \boldsymbol{n}')$ with $\boldsymbol{n}'=(4,0,7)$")
ax[0].imshow(local_information_spread[:, :, 7].T, cmap="bwr", vmin=-.002, vmax=.002, origin="lower")
ax[0].set_title(r'$n_z=7$')
ax[0].set_xlabel(r'$n_x$')
ax[0].set_ylabel(r'$n_y$')
ax[1].imshow(local_information_spread[:, 0, :].T, cmap="bwr", vmin=-.002, vmax=.002, origin="lower")
ax[1].set_title(r'$n_y=0$')
ax[1].set_xlabel(r'$n_x$')
ax[1].set_ylabel(r'$n_z$')
ax[2].imshow(local_information_spread[4, :, :].T, cmap="bwr", vmin=-.002, vmax=.002, origin="lower")
ax[2].set_title(r'$n_x=4$')
ax[2].set_xlabel(r'$n_y$')
ax[2].set_ylabel(r'$n_z$')
plt.tight_layout()
plt.savefig("figures/i_selections_15.png", dpi=300)
plt.show()
