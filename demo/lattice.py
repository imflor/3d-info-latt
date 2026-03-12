import numpy as np


class InformationLattice:
    """
    Information–lattice computation for rectangular subsystems on a square lattice.
    Includes parallel computation of von Neumann information values.

    Parameters
    ----------
    n_sites : tuple[int, int] | list[int]
        System size (Nx, Ny).
    parallel : bool, optional
        If true (default), uses parallel computation using the `joblib` library.
    loader : bool, optional
        If true (default), uses the progress bar from the `tqdm` library.

    Attributes
    ----------
    Nx, Ny : int
        Number of physical sites along each direction.
    n : int
        Total number of physical sites (Nx*Ny).
    physical_lattice : np.ndarray, shape (Nx, Ny)
        Integer physical site labels on the lattice.
    subsystems_lattice : np.ndarray, shape (Nx, Ny, Nx, Ny, Nx*Ny)
        Lattice of all subsystems: subsystem sites for (lx, ly, nx, ny) are stored in
        subsystems_lattice[lx, ly, nx, ny, :((lx+1)*(ly+1))].
    i_vn : np.ndarray, shape (Nx+2, Ny+2, Nx+1, Ny+1)
        Von Neumann information I_vN(C) for each subsystem; extra columns/rows added for well-defined indexing.
    i_local : np.ndarray, shape (Nx, Ny, Nx, Ny)
        Local information from inclusion–exclusion, indexed as i_local[lx, ly, nx, ny].

    """

    def __init__(self, n_sites, parallel=True, loader=True):
        self.n_sites = np.array(n_sites, dtype=int)
        self.Nx, self.Ny = map(int, self.n_sites)
        self.n = int(self.n_sites.prod())
        self.parallel = parallel
        self.loader = loader

        self.physical_lattice = np.arange(self.n).reshape(self.Nx, self.Ny)

        # Generate all rectangular subsystems C^(n,l).
        self.subsystems_lattice = self._generate_subsystems_lattice()

        # Von Neumann information and local information on each subssytem
        self.i_vn = np.zeros((self.Nx + 2, self.Ny + 2, self.Nx + 1, self.Ny + 1))
        self.i_local = np.zeros((self.Nx, self.Ny, self.Nx, self.Ny))

    def compute(self, state):
        """
        Given a `state`, computes its information lattice.

        Parameters
        ----------
        state : State
            Object representing the quantum state. Contains the method `entanglement_entropy`.
        """
        self.compute_von_neumann_information(state)
        self.compute_local_information()

    def _get_subsystem_sites(self, n, l):
        """
        Return physical site indices for the rectangle C^(n,l).
        n = (nx, ny) is the lower-left corner and l = (lx, ly) is the scale,
        so the rectangle contains (lx+1) × (ly+1) physical sites.
        """
        nx, ny = map(int, n)
        lx, ly = map(int, l)
        xs = range(nx, nx + lx + 1)
        ys = range(ny, ny + ly + 1)
        return self.physical_lattice[np.ix_(xs, ys)].reshape(-1)

    def _generate_subsystems_lattice(self):
        """
        Organizes all subsystems C^(n,l) into an array.
        """
        subsystems_lattice = np.zeros((self.Nx, self.Ny, self.Nx, self.Ny, self.Nx * self.Ny), dtype=int)
        for lx in range(self.Nx):
            for ly in range(self.Ny):
                n_sites = (lx + 1) * (ly + 1)
                for nx in range(self.Nx - lx):
                    for ny in range(self.Ny - ly):
                        subsystems_lattice[lx, ly, nx, ny, :n_sites] = \
                            self._get_subsystem_sites((nx, ny), (lx, ly))
        return subsystems_lattice

    def compute_von_neumann_information(self, state, batch_size=25):
        """
        Fill self.i_vn with the Von Neumann information for all rectangles.
        The quantity stored is: I_vN(C) = |C| - S(C),
        where S(C) is the entanglement entropy of subsystem C in the given state.
        """

        def i_vn_function(lx, ly, nx, ny):
            n_sites = (lx + 1) * (ly + 1)
            sites = self.subsystems_lattice[lx, ly, nx, ny, :n_sites]
            return n_sites - state.entanglement_entropy(sites)

        jobs = [
            (lx, ly, nx, ny)
            for lx in range(self.Nx)
            for ly in range(self.Ny)
            for nx in range(self.Nx - lx)
            for ny in range(self.Ny - ly)
        ]

        from utils.parallel import map_jobs

        for job, val in map_jobs(jobs, i_vn_function, parallel=self.parallel, batch_size=batch_size, loader=True):
            lx, ly, nx, ny = job
            self.i_vn[lx, ly, nx, ny] = val

    def compute_local_information(self):
        """Compute the local information on the full (n, l) lattice using inclusion–exclusion."""
        for lx in range(self.Nx):
            for ly in range(self.Ny):
                for nx in range(self.Nx - lx):
                    for ny in range(self.Ny - ly):
                        self.i_local[lx, ly, nx, ny] = self._local_information(lx, ly, nx, ny)

    def _local_information(self, lx, ly, nx, ny):
        """
        Inclusion--exclusion local information at any given (lx, ly, nx, ny).

        Notes
        -----
        self.i_vn has shape (Nx+2, Ny+2, Nx+1, Ny+1). Indices like (lx-1) or (ly-2)
        may become negative at small scales; the additional columns and rows ensure these return zero.

        """
        i = self.i_vn
        return i[lx, ly, nx, ny] \
            - i[lx - 1, ly, nx, ny] - i[lx - 1, ly, nx + 1, ny] \
            - i[lx, ly - 1, nx, ny] - i[lx, ly - 1, nx, ny + 1] \
            + i[lx - 2, ly, nx + 1, ny] \
            + i[lx, ly - 2, nx, ny + 1] \
            + i[lx - 1, ly - 1, nx, ny] + i[lx - 1, ly - 1, nx + 1, ny + 1] \
            + i[lx - 1, ly - 1, nx + 1, ny] + i[lx - 1, ly - 1, nx, ny + 1] \
            - i[lx - 2, ly - 1, nx + 1, ny] - i[lx - 2, ly - 1, nx + 1, ny + 1] \
            - i[lx - 1, ly - 2, nx, ny + 1] - i[lx - 1, ly - 2, nx + 1, ny + 1] \
            + i[lx - 2, ly - 2, nx + 1, ny + 1]

