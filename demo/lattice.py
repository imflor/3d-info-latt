import numpy as np


class InformationLattice:
    """
    Information-lattice computation for cuboid subsystems on a 3D lattice.
    Includes parallel computation of von Neumann information values.

    Parameters
    ----------
    n_sites : tuple[int, int, int] | list[int]
        System size (Nx, Ny, Nz).
    parallel : bool, optional
        If true (default), uses parallel computation using the `joblib` library.
    loader : bool, optional
        If true (default), uses the progress bar from the `tqdm` library.

    Attributes
    ----------
    Nx, Ny, Nz : int
        Number of physical sites along each direction.
    n : int
        Total number of physical sites (Nx*Ny*Nz).
    physical_lattice : np.ndarray, shape (Nx, Ny, Nz)
        Integer physical site labels on the lattice.
    subsystems_lattice : np.ndarray, shape (Nx, Ny, Nz, Nx, Ny, Nz, Nx*Ny*Nz)
        Lattice of all cuboid subsystems: subsystem sites for (lx, ly, lz, nx, ny, nz) are stored in
        subsystems_lattice[lx, ly, lz, nx, ny, nz, :((lx+1)*(ly+1)*(lz+1))].
    i_vn : np.ndarray, shape (Nx+2, Ny+2, Nz+2, Nx+1, Ny+1, Nz+1)
        Von Neumann information I_vN(C) for each subsystem; extra hyperplanes added for well-defined indexing.
    i_local : np.ndarray, shape (Nx, Ny, Nz, Nx, Ny, Nz)
        Local information from inclusion-exclusion, indexed as i_local[lx, ly, lz, nx, ny, nz].

    """

    def __init__(self, n_sites, parallel=True, loader=True):
        self.n_sites = np.array(n_sites, dtype=int)
        self.Nx, self.Ny, self.Nz = map(int, self.n_sites)
        self.n = int(self.n_sites.prod())
        self.parallel = parallel
        self.loader = loader

        self.physical_lattice = np.arange(self.n).reshape(self.Nx, self.Ny, self.Nz)

        # Generate all cuboid subsystems C^(n,l).
        self.subsystems_lattice = self._generate_subsystems_lattice()

        # Von Neumann information and local information on each subsystem.
        self.i_vn = np.zeros((self.Nx + 2, self.Ny + 2, self.Nz + 2, self.Nx + 1, self.Ny + 1, self.Nz + 1))
        self.i_local = np.zeros((self.Nx, self.Ny, self.Nz, self.Nx, self.Ny, self.Nz))

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
        Return physical site indices for the cuboid C^(n,l).
        n = (nx, ny, nz) is one corner and l = (lx, ly, lz) is the scale,
        so the cuboid contains (lx+1) x (ly+1) x (lz+1) physical sites.
        """
        nx, ny, nz = map(int, n)
        lx, ly, lz = map(int, l)
        xs = range(nx, nx + lx + 1)
        ys = range(ny, ny + ly + 1)
        zs = range(nz, nz + lz + 1)
        return self.physical_lattice[np.ix_(xs, ys, zs)].reshape(-1)

    def _generate_subsystems_lattice(self):
        """
        Organizes all subsystems C^(n,l) into an array.
        """
        subsystems_lattice = np.zeros(
            (self.Nx, self.Ny, self.Nz, self.Nx, self.Ny, self.Nz, self.n),
            dtype=int,
        )
        for lx in range(self.Nx):
            for ly in range(self.Ny):
                for lz in range(self.Nz):
                    n_sites = (lx + 1) * (ly + 1) * (lz + 1)
                    for nx in range(self.Nx - lx):
                        for ny in range(self.Ny - ly):
                            for nz in range(self.Nz - lz):
                                subsystems_lattice[lx, ly, lz, nx, ny, nz, :n_sites] = (
                                    self._get_subsystem_sites((nx, ny, nz), (lx, ly, lz))
                                )
        return subsystems_lattice

    def compute_von_neumann_information(self, state, batch_size=25):
        """
        Fill self.i_vn with the Von Neumann information for all cuboids.
        The quantity stored is: I_vN(C) = |C| - S(C),
        where S(C) is the entanglement entropy of subsystem C in the given state.
        """

        def i_vn_function(lx, ly, lz, nx, ny, nz):
            n_sites = (lx + 1) * (ly + 1) * (lz + 1)
            sites = self.subsystems_lattice[lx, ly, lz, nx, ny, nz, :n_sites]
            return n_sites - state.entanglement_entropy(sites)

        jobs = [
            (lx, ly, lz, nx, ny, nz)
            for lx in range(self.Nx)
            for ly in range(self.Ny)
            for lz in range(self.Nz)
            for nx in range(self.Nx - lx)
            for ny in range(self.Ny - ly)
            for nz in range(self.Nz - lz)
        ]

        from utils.parallel import map_jobs

        for job, val in map_jobs(
            jobs,
            i_vn_function,
            parallel=self.parallel,
            batch_size=batch_size,
            loader=self.loader,
        ):
            lx, ly, lz, nx, ny, nz = job
            self.i_vn[lx, ly, lz, nx, ny, nz] = val

    def compute_local_information(self):
        """Compute the local information on the full (n, l) lattice using inclusion–exclusion."""
        for lx in range(self.Nx):
            for ly in range(self.Ny):
                for lz in range(self.Nz):
                    for nx in range(self.Nx - lx):
                        for ny in range(self.Ny - ly):
                            for nz in range(self.Nz - lz):
                                self.i_local[lx, ly, lz, nx, ny, nz] = self._local_information(
                                    lx,
                                    ly,
                                    lz,
                                    nx,
                                    ny,
                                    nz,
                                )

    def _local_information(self, lx, ly, lz, nx, ny, nz):
        """
        Inclusion--exclusion local information at any given (lx, ly, lz, nx, ny, nz).

        Notes
        -----
        self.i_vn has shape (Nx+2, Ny+2, Nz+2, Nx+1, Ny+1, Nz+1). Indices like
        (lx-1), (ly-2), or (lz-2) may become negative at small scales; the additional
        hyperplanes ensure these return zero.
        """
        i = self.i_vn
        return (
            i[lx, ly, lz, nx, ny, nz]
            - i[lx - 1, ly, lz, nx, ny, nz]
            - i[lx - 1, ly, lz, nx + 1, ny, nz]
            - i[lx, ly - 1, lz, nx, ny, nz]
            - i[lx, ly - 1, lz, nx, ny + 1, nz]
            - i[lx, ly, lz - 1, nx, ny, nz]
            - i[lx, ly, lz - 1, nx, ny, nz + 1]
            + i[lx - 1, ly - 1, lz, nx, ny, nz]
            + i[lx - 1, ly - 1, lz, nx + 1, ny + 1, nz]
            + i[lx - 1, ly - 1, lz, nx + 1, ny, nz]
            + i[lx - 1, ly - 1, lz, nx, ny + 1, nz]
            + i[lx - 1, ly, lz - 1, nx, ny, nz]
            + i[lx - 1, ly, lz - 1, nx + 1, ny, nz + 1]
            + i[lx - 1, ly, lz - 1, nx + 1, ny, nz]
            + i[lx - 1, ly, lz - 1, nx, ny, nz + 1]
            + i[lx, ly - 1, lz - 1, nx, ny, nz]
            + i[lx, ly - 1, lz - 1, nx, ny + 1, nz + 1]
            + i[lx, ly - 1, lz - 1, nx, ny + 1, nz]
            + i[lx, ly - 1, lz - 1, nx, ny, nz + 1]
            + i[lx - 2, ly, lz, nx + 1, ny, nz]
            + i[lx, ly - 2, lz, nx, ny + 1, nz]
            + i[lx, ly, lz - 2, nx, ny, nz + 1]
            - i[lx - 1, ly - 1, lz - 1, nx, ny, nz]
            - i[lx - 1, ly - 1, lz - 1, nx + 1, ny + 1, nz + 1]
            - i[lx - 1, ly - 1, lz - 1, nx + 1, ny, nz]
            - i[lx - 1, ly - 1, lz - 1, nx, ny + 1, nz]
            - i[lx - 1, ly - 1, lz - 1, nx, ny, nz + 1]
            - i[lx - 1, ly - 1, lz - 1, nx + 1, ny + 1, nz]
            - i[lx - 1, ly - 1, lz - 1, nx + 1, ny, nz + 1]
            - i[lx - 1, ly - 1, lz - 1, nx, ny + 1, nz + 1]
            - i[lx - 2, ly - 1, lz, nx + 1, ny, nz]
            - i[lx - 2, ly - 1, lz, nx + 1, ny + 1, nz]
            - i[lx - 2, ly, lz - 1, nx + 1, ny, nz]
            - i[lx - 2, ly, lz - 1, nx + 1, ny, nz + 1]
            - i[lx - 1, ly - 2, lz, nx, ny + 1, nz]
            - i[lx - 1, ly - 2, lz, nx + 1, ny + 1, nz]
            - i[lx, ly - 2, lz - 1, nx, ny + 1, nz]
            - i[lx, ly - 2, lz - 1, nx, ny + 1, nz + 1]
            - i[lx - 1, ly, lz - 2, nx, ny, nz + 1]
            - i[lx - 1, ly, lz - 2, nx + 1, ny, nz + 1]
            - i[lx, ly - 1, lz - 2, nx, ny, nz + 1]
            - i[lx, ly - 1, lz - 2, nx, ny + 1, nz + 1]
            + i[lx - 2, ly - 1, lz - 1, nx + 1, ny, nz]
            + i[lx - 2, ly - 1, lz - 1, nx + 1, ny + 1, nz]
            + i[lx - 2, ly - 1, lz - 1, nx + 1, ny, nz + 1]
            + i[lx - 2, ly - 1, lz - 1, nx + 1, ny + 1, nz + 1]
            + i[lx - 1, ly - 2, lz - 1, nx, ny + 1, nz]
            + i[lx - 1, ly - 2, lz - 1, nx + 1, ny + 1, nz]
            + i[lx - 1, ly - 2, lz - 1, nx, ny + 1, nz + 1]
            + i[lx - 1, ly - 2, lz - 1, nx + 1, ny + 1, nz + 1]
            + i[lx - 1, ly - 1, lz - 2, nx, ny, nz + 1]
            + i[lx - 1, ly - 1, lz - 2, nx + 1, ny, nz + 1]
            + i[lx - 1, ly - 1, lz - 2, nx, ny + 1, nz + 1]
            + i[lx - 1, ly - 1, lz - 2, nx + 1, ny + 1, nz + 1]
            + i[lx - 2, ly - 2, lz, nx + 1, ny + 1, nz]
            + i[lx - 2, ly, lz - 2, nx + 1, ny, nz + 1]
            + i[lx, ly - 2, lz - 2, nx, ny + 1, nz + 1]
            - i[lx - 2, ly - 2, lz - 1, nx + 1, ny + 1, nz]
            - i[lx - 2, ly - 2, lz - 1, nx + 1, ny + 1, nz + 1]
            - i[lx - 2, ly - 1, lz - 2, nx + 1, ny, nz + 1]
            - i[lx - 2, ly - 1, lz - 2, nx + 1, ny + 1, nz + 1]
            - i[lx - 1, ly - 2, lz - 2, nx, ny + 1, nz + 1]
            - i[lx - 1, ly - 2, lz - 2, nx + 1, ny + 1, nz + 1]
            + i[lx - 2, ly - 2, lz - 2, nx + 1, ny + 1, nz + 1]
        )
