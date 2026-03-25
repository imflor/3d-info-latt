import matplotlib.pyplot as plt
import infolattice as il


n_sites = (20, 20, 3)
state = il.TightBindingGS(n_sites, t=1)
lat = il.InformationLattice(n_sites, parallel=True, loader=True)
lat.compute(state)


## Plot the information per multi-scale

information_per_scale = lat.i_local.sum(axis=(3, 4, 5))
fig, ax = il.plot_3d_array(
    information_per_scale,
    vmax=0.7,
    cutoff=0.01,
    power=0.5,
    max_alpha=0.95,
    marker_scale=180,
    view=(20, -55),
    marker="s",
)
plt.show()


## Plot the local information spread

reference_point = (5, 5, 0)
local_information_spread = lat.spread_around_point(*reference_point)
il.save_rotating_3d_array(
    local_information_spread,
    save_path="figures/i_centered.gif",
    vmax=0.005,
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