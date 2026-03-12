from pathlib import Path
import sys
import numpy as np

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


def sample_entries(array, max_entries=5, tol=1e-9):
    """Return up to `max_entries` entries with the largest magnitude above `tol`."""
    coords = np.argwhere(np.abs(array) > tol)
    if coords.shape[0] == 0:
        return []
    values = np.abs(array[tuple(coords.T)])
    order = np.argsort(values)[::-1]
    return [
        (tuple(int(v) for v in coords[idx]), float(array[tuple(coords[idx])]))
        for idx in order[:max_entries]
    ]


# Initialize state and wave function: randomly paired singlets on a 3D lattice.
n_qubits = [3, 3, 2]
seed = 0
state_vector, pairs = physics.random_singlets(n_qubits, seed)

# State object (contains entanglement entropy function)
state = physics.State(state_vector)

# Compute the information lattice for this state
information_lattice = lattice.InformationLattice(n_qubits, parallel=ENABLE_PARALLEL, loader=ENABLE_LOADER)
information_lattice.compute(state)

Nx, Ny, Nz = map(int, n_qubits)
valid_i_vn = information_lattice.i_vn[:Nx, :Ny, :Nz, :Nx, :Ny, :Nz]

print(f"3D random-singlets lattice computed for n_sites={(Nx, Ny, Nz)}")
print(f"i_vn shape: {information_lattice.i_vn.shape}")
print(f"i_local shape: {information_lattice.i_local.shape}")
print(f"nonzero cuboid information values: {np.count_nonzero(np.abs(valid_i_vn) > 1e-9)}")
print(f"Total information: {np.sum(information_lattice.i_local)}")

free_fermion_state = physics.TightBindingGS(n_qubits, t=1)
free_fermion_subset = np.arange(free_fermion_state.n).reshape(Nx, Ny, Nz)[:2, :2, :1].reshape(-1)
free_fermion_entropy = free_fermion_state.entanglement_entropy(free_fermion_subset)
information_lattice.compute(free_fermion_state)

print("\n3D free-fermion test:")
print(f"subset sites: {free_fermion_subset.tolist()}")
print(f"entanglement entropy: {free_fermion_entropy:.12f}")
print(f"Total information: {np.sum(information_lattice.i_local)}")
