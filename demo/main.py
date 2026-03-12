import physics
import lattice
import matplotlib.pyplot as plt
import utils.plotting as plot


ENABLE_PARALLEL = True
ENABLE_LOADER = True

# Initialize state and wave function: randomly paired singlets on a 2D lattice
n_qubits = [4, 4]
seed = 0
state_vector, _ = physics.random_singlets(n_qubits, seed)

# State object (contains entanglement entropy function)
state = physics.State(state_vector)

# Compute the information lattice for this state
information_lattice = lattice.InformationLattice(n_qubits, parallel=ENABLE_PARALLEL, loader=ENABLE_LOADER)
information_lattice.compute(state)

# Plot the information lattice
fig, ax_map = plot.plot_infolattice(
    i_local=information_lattice.i_local,
    colors=['w', 'r', 'k']  # add colors if necessary
)
plt.savefig("random_singlets.pdf", bbox_inches="tight")
plt.show()
