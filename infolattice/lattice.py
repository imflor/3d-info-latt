import numpy as np

from .parallel import map_jobs


class InformationLattice:
    """
    Information-lattice computation for cuboid subsystems on a 3D lattice.
    Includes parallel computation of von Neumann information values.
    """

    def __init__(self, n_sites, parallel=True, loader=True):
        self.n_sites = np.array(n_sites, dtype=int)
        self.Nx, self.Ny, self.Nz = map(int, self.n_sites)
        self.n = int(self.n_sites.prod())
        self.parallel = parallel
        self.loader = loader

        self.physical_lattice = np.arange(self.n).reshape(self.Nx, self.Ny, self.Nz)
        self.subsystems_lattice = self._generate_subsystems_lattice()
        self.i_vn = np.zeros((self.Nx + 2, self.Ny + 2, self.Nz + 2, self.Nx + 1, self.Ny + 1, self.Nz + 1))
        self.i_local = np.zeros((self.Nx, self.Ny, self.Nz, self.Nx, self.Ny, self.Nz))

    def compute(self, state):
        self.compute_von_neumann_information(state)
        self.compute_local_information()

    def _get_subsystem_sites(self, n, l):
        nx, ny, nz = map(int, n)
        lx, ly, lz = map(int, l)
        xs = range(nx, nx + lx + 1)
        ys = range(ny, ny + ly + 1)
        zs = range(nz, nz + lz + 1)
        return self.physical_lattice[np.ix_(xs, ys, zs)].reshape(-1)

    def _generate_subsystems_lattice(self):
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

    def _spread_from_corner(self, corner, x, y, z, fill=np.nan, crop=True):
        Lx, Ly, Lz = np.meshgrid(
            np.arange(self.Nx),
            np.arange(self.Ny),
            np.arange(self.Nz),
            indexing="ij",
        )

        y_side, x_side, z_side = corner
        A = x - (x_side == "r") * Lx
        B = y - (y_side == "t") * Ly
        C = z - (z_side == "b") * Lz

        valid = (
            (A >= 0) & (B >= 0) & (C >= 0)
            & (A < (self.Nx - Lx))
            & (B < (self.Ny - Ly))
            & (C < (self.Nz - Lz))
        )

        if crop:
            ix = np.where(valid.any(axis=(1, 2)))[0]
            iy = np.where(valid.any(axis=(0, 2)))[0]
            iz = np.where(valid.any(axis=(0, 1)))[0]
            I = np.ix_(ix, iy, iz)
            out = self.i_local[Lx[I], Ly[I], Lz[I], A[I], B[I], C[I]]
        else:
            out = np.full((self.Nx, self.Ny, self.Nz), fill, dtype=float)
            out[valid] = self.i_local[Lx[valid], Ly[valid], Lz[valid], A[valid], B[valid], C[valid]]

        if x_side == "r":
            out = out[::-1, :, :]
        if y_side == "t":
            out = out[:, ::-1, :]
        if z_side == "b":
            out = out[:, :, ::-1]
        return out

    def spread_around_point(self, x, y, z, fill=np.nan):
        if not (0 <= x < self.Nx and 0 <= y < self.Ny and 0 <= z < self.Nz):
            raise IndexError("Reference point is outside the lattice.")

        blocks = {
            "blf": self._spread_from_corner("blf", x, y, z, fill=fill, crop=True),
            "brf": self._spread_from_corner("brf", x, y, z, fill=fill, crop=True),
            "tlf": self._spread_from_corner("tlf", x, y, z, fill=fill, crop=True),
            "trf": self._spread_from_corner("trf", x, y, z, fill=fill, crop=True),
            "blb": self._spread_from_corner("blb", x, y, z, fill=fill, crop=True),
            "brb": self._spread_from_corner("brb", x, y, z, fill=fill, crop=True),
            "tlb": self._spread_from_corner("tlb", x, y, z, fill=fill, crop=True),
            "trb": self._spread_from_corner("trb", x, y, z, fill=fill, crop=True),
        }

        out = np.full(self.i_local.shape[:3], fill, dtype=float)
        x_size, y_size, z_size = blocks["trb"].shape

        for corner, block in blocks.items():
            y_side, x_side, z_side = corner
            xs = slice(None, x_size) if x_side == "r" else slice(x_size - 1, None)
            ys = slice(None, y_size) if y_side == "t" else slice(y_size - 1, None)
            zs = slice(None, z_size) if z_side == "b" else slice(z_size - 1, None)
            out[xs, ys, zs] = block

        return out
