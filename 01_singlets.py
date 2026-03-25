import numpy as np
import infolattice as il


n_sites = (3, 3, 2)
seed = 0

state_vector, pairs = il.random_singlets(n_sites, seed=seed)
state = il.State(state_vector)

lat = il.InformationLattice(n_sites, parallel=True, loader=True)
lat.compute(state)

print(f"random singlets on n_sites={n_sites}")
print(f"number of singlet pairs: {len(pairs)}")
print(f"i_vn shape: {lat.i_vn.shape}")
print(f"i_local shape: {lat.i_local.shape}")
print(f"sum(i_local): {np.sum(lat.i_local):.12f}")
