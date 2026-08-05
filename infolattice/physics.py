from pathlib import Path

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
    pairs = [(int(p[2 * k]), int(p[2 * k + 1])) for k in range(n // 2)]
    s = np.array([0, 1, -1, 0], complex) / np.sqrt(2)
    psi = s
    for _ in range(n // 2 - 1):
        psi = np.kron(psi, s)
    inv = np.empty(n, int)
    inv[p] = np.arange(n)
    return psi.reshape((2,) * n).transpose(inv).reshape(2 ** n), pairs


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
        psi = np.reshape(psi, n * [2])
        psi = np.moveaxis(psi, subsystem, range(0, len_subsystem))
        psi = np.reshape(psi, (2 ** len_subsystem, -1))
        return psi


class TightBindingGS:

    def __init__(self, n_sites, t=1, correlation_path=None):
        self.tol_log = 1e-16
        self.n_sites = np.array(n_sites, dtype=int)
        self.nx, self.ny, self.nz = map(int, self.n_sites)
        self.n = int(self.n_sites.prod())
        self.periodic = False
        self.t = t
        if correlation_path is None:
            self.h = self.hamiltonian()
            self.e, self.v = self.diagonalize_hamiltonian()
            self.chi = self.correlations()
        else:
            self.h = None
            self.e = None
            self.v = None
            self.chi = self._load_correlation_matrix(correlation_path)

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
        v_occ = self.v[:, self.e <= 0]
        return v_occ @ v_occ.conj().T

    def correlation_matrix_shape(self):
        return self.n, self.n

    def _load_correlation_matrix(self, correlation_path):
        chi = np.load(correlation_path, mmap_mode="r")
        if chi.shape != self.correlation_matrix_shape():
            raise ValueError("Saved correlation matrix shape does not match n_sites.")
        return chi

    def save_correlation_matrix(self, correlation_path):
        correlation_path = Path(correlation_path)
        correlation_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(correlation_path, np.asarray(self.chi))


class NodalLineGS(TightBindingGS):

    def __init__(self, n_sites, periodic=False, m=2.8, v=1.0, surface_mass=0.0, correlation_path=None):
        self.tol_log = 1e-16
        self.n_sites = np.array(n_sites, dtype=int)
        self.nx, self.ny, self.nz = map(int, self.n_sites)
        self.n = int(self.n_sites.prod())
        self.periodic = bool(periodic)
        self.mass = float(m)
        self.v_orbital = float(v)
        self.surface_mass = float(surface_mass)
        if correlation_path is None:
            self.h = self.hamiltonian()
            self.e, self.v = self.diagonalize_hamiltonian()
            self.chi = self.correlations()
        else:
            self.h = None
            self.e = None
            self.v = None
            self.chi = self._load_correlation_matrix(correlation_path)

    def entanglement_entropy(self, subset):
        sites = np.asarray(subset, dtype=int)
        orbitals = np.empty(2 * len(sites), dtype=int)
        orbitals[0::2] = 2 * sites
        orbitals[1::2] = 2 * sites + 1
        C = self.chi[np.ix_(orbitals, orbitals)]
        s = np.linalg.eigvalsh(C)
        S = entropy_stable(s, self.tol_log)
        Sp = entropy_stable(1 - s, self.tol_log)
        return S.sum() + Sp.sum()

    def hamiltonian(self, periodic=None, m=None, v=None, surface_mass=None):
        periodic = self.periodic if periodic is None else bool(periodic)
        m = self.mass if m is None else float(m)
        v = self.v_orbital if v is None else float(v)
        surface_mass = self.surface_mass if surface_mass is None else float(surface_mass)

        nx, ny, nz = self.nx, self.ny, self.nz
        H = np.zeros((nx, ny, nz, 2, nx, ny, nz, 2), dtype=float)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    H[i, j, k, 0, i, j, k, 0] += +m
                    H[i, j, k, 1, i, j, k, 1] += -m

                    if surface_mass != 0.0:
                        on_boundary = (
                            i == 0 or i == nx - 1
                            or j == 0 or j == ny - 1
                            or k == 0 or k == nz - 1
                        )
                        if on_boundary:
                            H[i, j, k, 0, i, j, k, 1] += surface_mass
                            H[i, j, k, 1, i, j, k, 0] += surface_mass

                    ip = i + 1
                    if ip < nx or periodic:
                        ip %= nx
                        H[i, j, k, 0, ip, j, k, 0] += -0.5
                        H[ip, j, k, 0, i, j, k, 0] += -0.5

                        H[i, j, k, 1, ip, j, k, 1] += +0.5
                        H[ip, j, k, 1, i, j, k, 1] += +0.5

                    jp = j + 1
                    if jp < ny or periodic:
                        jp %= ny
                        H[i, j, k, 0, i, jp, k, 0] += -0.5
                        H[i, jp, k, 0, i, j, k, 0] += -0.5

                        H[i, j, k, 1, i, jp, k, 1] += +0.5
                        H[i, jp, k, 1, i, j, k, 1] += +0.5

                    kp = k + 1
                    if kp < nz or periodic:
                        kp %= nz

                        H[i, j, k, 0, i, j, kp, 0] += -0.5
                        H[i, j, kp, 0, i, j, k, 0] += -0.5

                        H[i, j, k, 1, i, j, kp, 1] += +0.5
                        H[i, j, kp, 1, i, j, k, 1] += +0.5

                        H[i, j, k, 0, i, j, kp, 1] += -v / 2.0
                        H[i, j, kp, 1, i, j, k, 0] += -v / 2.0

                        H[i, j, k, 1, i, j, kp, 0] += +v / 2.0
                        H[i, j, kp, 0, i, j, k, 1] += +v / 2.0

        return H.reshape(2 * self.n, 2 * self.n)

    def correlation_matrix_shape(self):
        return 2 * self.n, 2 * self.n


class PiFluxGS(TightBindingGS):

    def __init__(self, n_sites, t=1.0, periodic=False, correlation_path=None):
        self.tol_log = 1e-16
        self.n_sites = np.array(n_sites, dtype=int)
        self.nx, self.ny, self.nz = map(int, self.n_sites)
        self.n = int(self.n_sites.prod())
        self.t = float(t)
        self.periodic = bool(periodic)
        if correlation_path is None:
            self.h = self.hamiltonian()
            self.e, self.v = self.diagonalize_hamiltonian()
            self.chi = self.correlations()
        else:
            self.h = None
            self.e = None
            self.v = None
            self.chi = self._load_correlation_matrix(correlation_path)

    def hamiltonian(self, periodic=None, t=None):
        periodic = self.periodic if periodic is None else bool(periodic)
        t = self.t if t is None else float(t)

        H = np.zeros((self.nx, self.ny, self.nz, self.nx, self.ny, self.nz), dtype=float)
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    ip = i + 1
                    if ip < self.nx or periodic:
                        ip %= self.nx
                        H[i, j, k, ip, j, k] = H[ip, j, k, i, j, k] = -t

                    jp = j + 1
                    if jp < self.ny or periodic:
                        jp %= self.ny
                        ty = t * ((-1.0) ** i)
                        H[i, j, k, i, jp, k] = H[i, jp, k, i, j, k] = -ty

                    kp = k + 1
                    if kp < self.nz or periodic:
                        kp %= self.nz
                        tz = t * ((-1.0) ** (i + j))
                        H[i, j, k, i, j, kp] = H[i, j, kp, i, j, k] = -tz

        return H.reshape(self.n, self.n)
