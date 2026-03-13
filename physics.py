import numpy as np


def entropy_stable(s, tol=1e-15):
    s_clipped = np.clip(s, tol, 1 - tol)
    entropy = - s_clipped * np.log2(s_clipped)
    return entropy


def random_singlets(n_qubits, seed=None):
    """Produces a state with randomly paired qubits on a lattice of shape `n_qubits`."""
    n = int(np.prod(n_qubits))
    if n % 2:
        raise ValueError("n must be even")
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    pairs = [(int(p[2*k]), int(p[2*k+1])) for k in range(n//2)]
    s = np.array([0, 1, -1, 0], complex) / np.sqrt(2)
    psi = s
    for _ in range(n//2 - 1):
        psi = np.kron(psi, s)
    inv = np.empty(n, int)
    inv[p] = np.arange(n)
    return psi.reshape((2,)*n).transpose(inv).reshape(2**n), pairs


class State:
    """
    Parameters
    ----------
    state_vector : np.ndarray
        2^N state vector on N qubits.

    Methods
    ----------
    entanglement_entropy: (subsystem) -> float
        Entanglement entropy of the subsystem from the Schmidt decomposition of `state_vector`.
        This function can be replaced by alternative methods for calculating subsystem entropies
        (e.g., efficient calculation in free fermion states or an analytic expression of the entropy).
    """

    def __init__(self, state_vector):
        self.psi = state_vector

    def entanglement_entropy(self, subsystem):
        """Given the subsystem site indices, compute the entanglement entropy from the Schmidt values."""
        psi = self._reshape_psi(self.psi, subsystem)
        sv = np.linalg.svd(psi, compute_uv=False)
        while np.where(np.isnan(sv))[0].shape[0] > 0:
            psi += 1e-16 * np.random.random(psi.shape)
            psi /= np.linalg.norm(psi)
            sv = np.linalg.svd(psi, compute_uv=False)
        return entropy_stable(sv ** 2).sum()

    @staticmethod
    def _reshape_psi(psi, subsystem):
        """Reshape `psi` into the two sectors of the Schmidt decomposition."""
        len_subsystem = subsystem.shape[0]
        n = int(np.log2(np.prod(psi.shape)))
        # Bi-partition of Hilbert space
        psi = np.reshape(psi, n * [2])
        psi = np.moveaxis(psi, subsystem, range(0, len_subsystem))
        psi = np.reshape(psi, (2 ** len_subsystem, -1))
        return psi


class TightBindingGS:

    def __init__(self, n_sites, t=1):
        self.tol_log = 1e-16
        self.n_sites = np.array(n_sites, dtype=int)
        self.nx, self.ny, self.nz = map(int, self.n_sites)
        self.n = int(self.n_sites.prod())
        self.t = t
        self.h = self.hamiltonian()
        self.e, self.v = self.diagonalize_hamiltonian()
        self.chi = self.correlations()

    def entanglement_entropy(self, subset):
        i = np.asarray(subset, dtype=int)
        C = self.chi[np.ix_(i, i)]
        s = np.linalg.eigvalsh(C)
        S = entropy_stable(s, self.tol_log)
        Sp = entropy_stable(1 - s, self.tol_log)
        return S.sum() + Sp.sum()

    def hamiltonian(self):
        H = np.zeros((self.nx, self.ny, self.nz, self.nx, self.ny, self.nz), dtype=float)
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    if i + 1 < self.nx:
                        H[i, j, k, i + 1, j, k] = H[i + 1, j, k, i, j, k] = -self.t
                    if j + 1 < self.ny:
                        H[i, j, k, i, j + 1, k] = H[i, j + 1, k, i, j, k] = -self.t
                    if k + 1 < self.nz:
                        H[i, j, k, i, j, k + 1] = H[i, j, k + 1, i, j, k] = -self.t
        return H.reshape(self.n, self.n)

    def diagonalize_hamiltonian(self):
        return np.linalg.eigh(self.h)

    def correlations(self):
        occ = (self.e <= 0).astype(float)
        return self.v @ np.diag(occ) @ self.v.conj().T
