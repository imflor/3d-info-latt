import numpy as np


def entropy_stable(s, tol=1e-15):
    s_clipped = np.clip(s, tol, 1 - tol)
    entropy = - s_clipped * np.log2(s_clipped)
    return entropy


def random_singlets(n_qubits, seed=None):
    """ Produces a state with randomly paired qubits on a given number of qubits."""
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
        """ Given a wave function and an index set, calculates the Schmidt decomposition over A and B, with A the
        subsystem (a given set of indices) and B the complement. Calculates entropy from the singular values. """
        psi = self._reshape_psi(self.psi, subsystem)
        sv = np.linalg.svd(psi, compute_uv=False)
        while np.where(np.isnan(sv))[0].shape[0] > 0:
            psi += 1e-16 * np.random.random(psi.shape)
            psi /= np.linalg.norm(psi)
            sv = np.linalg.svd(psi, compute_uv=False)
        return entropy_stable(sv ** 2).sum()

    @staticmethod
    def _reshape_psi(psi, subsystem):
        """ Reshapes `psi` in two sectors A and B of the Schmidt decomposition. """
        len_subsystem = subsystem.shape[0]
        n = int(np.log2(np.prod(psi.shape)))
        # Bi-partition of Hilbert space
        psi = np.reshape(psi, n * [2])
        psi = np.moveaxis(psi, subsystem, range(0, len_subsystem))
        psi = np.reshape(psi, (2 ** len_subsystem, -1))
        return psi
