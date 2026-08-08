from pathlib import Path
import infolattice as il
import matplotlib.pyplot as plt

run_dir = Path("/Users/matthiasflor/Documents/1. PhD/3. Information lattice/3d-info-latt/cluster/runs/nodal_line_15x15x15")
run_dir = Path("/Users/matthiasflor/Documents/1. PhD/3. Information lattice/3d-info-latt/benchmark/runs/bench_nodal_line_L18_periodic")
run_dir = Path("/Users/matthiasflor/Documents/1. PhD/3. Information lattice/3d-info-latt/cluster/runs/nodal_line_17x17x17")
run_dir = Path("/Users/matthiasflor/Documents/1. PhD/3. Information lattice/3d-info-latt/cluster/runs/pi_flux_17x17x17")

manifest = run_dir / "manifest.json"

n_sites = (17, 17, 17)
lat = il.InformationLattice(n_sites, precompute_subsystems=False)
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

reference_point = (8, 8, 8)
local_information_spread = lat.spread_around_point(*reference_point)
il.save_rotating_3d_array(
    local_information_spread,
    save_path="figures/i_fermisurface_17.gif",
    vmax=.0008,
    cutoff=0.3,
    power=1.1,
    max_alpha=.9,
    marker_scale=100,
    marker="o",
    elev=30,
    azim_start=170,
    azim_stop=530,
    frames=120,
    fps=20,
)

##

reference_point = (8, 8, 8)
local_information_spread = lat.spread_around_point(*reference_point)
il.save_rotating_3d_array(
    local_information_spread,
    save_path="figures/i_nodal_17.gif",
    vmax=.0002,
    cutoff=0.1,
    power=.8,
    max_alpha=.9,
    marker_scale=100,
    marker="o",
    elev=30,
    azim_start=170,
    azim_stop=530,
    frames=120,
    fps=20,
)

##

reference_point = (8, 8, 8)
local_information_spread = lat.spread_around_point(*reference_point)
il.save_rotating_3d_array(
    local_information_spread,
    save_path="figures/i_dirac_17.gif",
    vmax=.0001,
    cutoff=0.1,
    power=.7,
    max_alpha=.9,
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
plt.savefig("figures/i_projections_nodal_15.png", dpi=300)
plt.show()

##

# IMPORTANT NOTE:
# MY FIRST THOUGHT WAS THAT THERE ARE NO REFLECTIONS
# BUT ACTUALLY, THEY ARE OUT OF THE PLANE SO WE DON'T SEE THEM HERE
# THE BEST WOULD BE ACTUALLY TO CONSIDER SLICES THAT ARE ORIENTED AT 45 DEG

# --> FUTURE TODO

x = reference_point

fig, ax = plt.subplots(1, 3, figsize=(9, 3))
fig.suptitle(r"$i(\boldsymbol{n}, \boldsymbol{n}')$ with $\boldsymbol{n}'=(4,0,7)$")
ax[0].imshow(local_information_spread[:, :, x[2]].T, cmap="bwr", vmin=-.00005, vmax=.00005, origin="lower")
ax[0].set_title(r'$n_z=7$')
ax[0].set_xlabel(r'$n_x$')
ax[0].set_ylabel(r'$n_y$')
ax[1].imshow(local_information_spread[:, x[1], :].T, cmap="bwr", vmin=-.00005, vmax=.00005, origin="lower")
ax[1].set_title(r'$n_y=0$')
ax[1].set_xlabel(r'$n_x$')
ax[1].set_ylabel(r'$n_z$')
ax[2].imshow(local_information_spread[x[0], :, :].T, cmap="bwr", vmin=-.00005, vmax=.00005, origin="lower")
ax[2].set_title(r'$n_x=4$')
ax[2].set_xlabel(r'$n_y$')
ax[2].set_ylabel(r'$n_z$')
plt.tight_layout()
plt.savefig("figures/i_selections_nodal_15.png", dpi=300)
plt.show()

## Information scaling

import numpy as np
xx = np.linspace(0, 15, 100)
plt.figure()
plt.loglog([lat.i_local[i, i, i, 0, 0, 0] for i in range(lat.Nx)])
plt.loglog(.0001 * xx**(-4))
plt.show()
