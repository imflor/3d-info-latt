from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

print(f"3D system size: Nx={Nx}, Ny={Ny}, Nz={Nz}")
print(f"Number of singlet pairs: {len(pairs)}")
print(f"First singlet pairs: {pairs[:5]}")
print(f"Entropy/information array shape (i_vn): {information_lattice.i_vn.shape}")
print(f"Local information array shape (i_local): {information_lattice.i_local.shape}")
print(f"Computed cuboid entropies: {np.count_nonzero(np.abs(valid_i_vn) > 1e-9)} nonzero entries")
print(f"Computed local values: {np.count_nonzero(np.abs(information_lattice.i_local) > 1e-9)} nonzero entries")

print("\nRepresentative i_vn entries:")
for index, value in sample_entries(valid_i_vn):
    print(f"  i_vn{index} = {value:.6f}")

print("\nRepresentative i_local entries:")
for index, value in sample_entries(information_lattice.i_local):
    print(f"  i_local{index} = {value:.6f}")
