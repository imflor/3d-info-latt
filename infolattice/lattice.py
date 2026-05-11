import json
from pathlib import Path

import numpy as np

from .parallel import greedy_chunk_indices, map_jobs, normalize_parallel


class InformationLattice:
    """
    Information-lattice computation for cuboid subsystems on a 3D lattice.
    Includes local and Slurm-oriented computation helpers.
    """

    def __init__(self, n_sites, parallel="joblib", loader=True, precompute_subsystems=True):
        self.n_sites = np.array(n_sites, dtype=int)
        self.Nx, self.Ny, self.Nz = map(int, self.n_sites)
        self.n = int(self.n_sites.prod())
        self.parallel = normalize_parallel(parallel)
        self.loader = loader
        self.batch_size = 25
        self.precompute_subsystems = bool(precompute_subsystems)

        self.physical_lattice = np.arange(self.n).reshape(self.Nx, self.Ny, self.Nz)
        self.subsystems_lattice = self._generate_subsystems_lattice() if self.precompute_subsystems else None
        self.i_vn = np.zeros((self.Nx + 2, self.Ny + 2, self.Nz + 2, self.Nx + 1, self.Ny + 1, self.Nz + 1))
        self.i_local = np.zeros((self.Nx, self.Ny, self.Nz, self.Nx, self.Ny, self.Nz))
        self._reset_i_vn()
        self._reset_i_local()

    def compute(self, state):
        self.compute_von_neumann_information(state, batch_size=self.batch_size)
        self.compute_local_information()

    def entropy_jobs(self):
        return [
            (lx, ly, lz, nx, ny, nz)
            for lx in range(self.Nx)
            for ly in range(self.Ny)
            for lz in range(self.Nz)
            for nx in range(self.Nx - lx)
            for ny in range(self.Ny - ly)
            for nz in range(self.Nz - lz)
        ]

    def entropy_job_array(self):
        jobs = self.entropy_jobs()
        if not jobs:
            return np.empty((0, 6), dtype=int)
        return np.asarray(jobs, dtype=int)

    def _expected_entropy_mask(self):
        mask = np.zeros((self.Nx, self.Ny, self.Nz, self.Nx, self.Ny, self.Nz), dtype=bool)
        for lx, ly, lz, nx, ny, nz in self.entropy_jobs():
            mask[lx, ly, lz, nx, ny, nz] = True
        return mask

    @staticmethod
    def _subsystem_volume(lx, ly, lz):
        return (int(lx) + 1) * (int(ly) + 1) * (int(lz) + 1)

    def _entropy_job_load(self, lx, ly, lz):
        """
        Heuristic load model for free-fermion entropies.

        The current entropy path diagonalizes the restricted correlation matrix on the
        literal cuboid sites, so the matrix dimension is the cuboid volume itself.
        We therefore estimate memory by m^2 and eigensolver time by m^3, and use
        greedy balancing with memory as the primary load and time as the tie-breaker.
        """
        m = self._subsystem_volume(lx, ly, lz)
        return m ** 2, m ** 3

    def _reset_i_vn(self):
        self.i_vn.fill(0.0)

    def _reset_i_local(self):
        self.i_local.fill(0.0)

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

    def _subsystem_sites(self, lx, ly, lz, nx, ny, nz):
        n_sites = (lx + 1) * (ly + 1) * (lz + 1)
        if self.subsystems_lattice is not None:
            return self.subsystems_lattice[lx, ly, lz, nx, ny, nz, :n_sites]
        return self._get_subsystem_sites((nx, ny, nz), (lx, ly, lz))

    def entropy_values_for_jobs(self, state, jobs):
        values = np.empty(len(jobs), dtype=float)
        for idx, (lx, ly, lz, nx, ny, nz) in enumerate(jobs):
            n_sites = (lx + 1) * (ly + 1) * (lz + 1)
            sites = self._subsystem_sites(lx, ly, lz, nx, ny, nz)
            values[idx] = n_sites - state.entanglement_entropy(sites)
        return values

    def compute_von_neumann_information(self, state, batch_size=None, jobs=None):
        batch_size = self.batch_size if batch_size is None else int(batch_size)
        jobs_given = jobs is not None
        jobs = self.entropy_jobs() if jobs is None else [tuple(map(int, job)) for job in jobs]

        if not jobs_given:
            self._reset_i_vn()

        if self.parallel == "slurm" and not jobs_given:
            raise RuntimeError(
                "parallel='slurm' prepares an external workflow only. "
                "Use cluster.prepare_tight_binding to write a manifest, "
                "cluster.worker to compute chunks, and cluster.assemble to rebuild the lattice."
            )

        parallel_mode = "none" if self.parallel == "slurm" else self.parallel

        def i_vn_function(lx, ly, lz, nx, ny, nz):
            n_sites = (lx + 1) * (ly + 1) * (lz + 1)
            sites = self._subsystem_sites(lx, ly, lz, nx, ny, nz)
            return n_sites - state.entanglement_entropy(sites)

        for job, val in map_jobs(
            jobs,
            i_vn_function,
            parallel=parallel_mode,
            batch_size=batch_size,
            loader=self.loader,
        ):
            lx, ly, lz, nx, ny, nz = job
            self.i_vn[lx, ly, lz, nx, ny, nz] = val

    def apply_entropy_results(self, jobs, values):
        jobs = np.asarray(jobs, dtype=int)
        values = np.asarray(values, dtype=float)
        if jobs.ndim != 2 or jobs.shape[1] != 6:
            raise ValueError("jobs must have shape (n_jobs, 6)")
        if values.shape[0] != jobs.shape[0]:
            raise ValueError("values must have the same length as jobs")
        self.i_vn[jobs[:, 0], jobs[:, 1], jobs[:, 2], jobs[:, 3], jobs[:, 4], jobs[:, 5]] = values

    def write_slurm_manifest(
        self,
        manifest_path,
        *,
        run_name=None,
        state_name,
        state_kwargs=None,
        state_path=None,
        n_chunks=1,
        shuffle_seed=0,
        output_name="lattice.npz",
    ):
        """Write a manifest and chunk layout for external Slurm execution."""
        manifest_path = Path(manifest_path)
        run_dir = manifest_path.parent
        data_dir = run_dir / "data"
        chunk_dir = data_dir / "chunks"
        run_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir.mkdir(parents=True, exist_ok=True)

        jobs = self.entropy_job_array()
        if shuffle_seed is not None:
            rng = np.random.default_rng(int(shuffle_seed))
            jobs = jobs[rng.permutation(len(jobs))]

        n_chunks = int(n_chunks)
        if n_chunks <= 0:
            raise ValueError("n_chunks must be positive")

        loads = [self._entropy_job_load(lx, ly, lz) for lx, ly, lz, _, _, _ in jobs]
        chunk_indices, chunk_loads = greedy_chunk_indices(loads, n_chunks)

        ordered_jobs = []
        chunks = []
        cursor = 0
        for chunk_id, job_indices in enumerate(chunk_indices):
            chunk_jobs = jobs[np.asarray(job_indices, dtype=int)] if job_indices else np.empty((0, 6), dtype=int)
            start = cursor
            stop = start + len(chunk_jobs)
            output = Path("data") / "chunks" / f"chunk_{chunk_id:05d}.npz"
            if len(chunk_jobs) > 0:
                ordered_jobs.append(chunk_jobs)
            chunks.append({
                "chunk_id": int(chunk_id),
                "start": int(start),
                "stop": int(stop),
                "n_jobs": int(len(chunk_jobs)),
                "estimated_memory_load": int(chunk_loads[chunk_id][0]),
                "estimated_time_load": int(chunk_loads[chunk_id][1]),
                "output": output.as_posix(),
            })
            cursor = stop

        jobs = np.concatenate(ordered_jobs, axis=0) if ordered_jobs else np.empty((0, 6), dtype=int)
        np.save(data_dir / "jobs.npy", jobs)

        manifest = {
            "version": 1,
            "run_name": None if run_name is None else str(run_name),
            "state": {
                "name": state_name,
                "kwargs": {} if state_kwargs is None else state_kwargs,
            },
            "lattice": {
                "kwargs": {
                    "n_sites": self.n_sites.tolist(),
                },
                "batch_size": int(self.batch_size),
                "parallel": "slurm",
            },
            "data": {
                "jobs": (Path("data") / "jobs.npy").as_posix(),
                "assembled": (Path("data") / output_name).as_posix(),
            },
            "chunking": {
                "strategy": "greedy_memory_time",
                "matrix_size": "subsystem_volume",
                "memory_load": "subsystem_volume**2",
                "time_load": "subsystem_volume**3",
            },
            "shuffle_seed": None if shuffle_seed is None else int(shuffle_seed),
            "n_jobs": int(len(jobs)),
            "n_chunks": int(len(chunks)),
            "chunks": chunks,
        }
        if state_path is not None:
            manifest["state"]["state_path"] = Path(state_path).as_posix()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    def load_slurm_results(self, manifest_path):
        """Load saved Slurm chunk outputs into this lattice and compute i_local."""
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())

        lattice_n_sites = np.array(manifest["lattice"]["kwargs"]["n_sites"], dtype=int)
        if not np.array_equal(lattice_n_sites, self.n_sites):
            raise ValueError("Manifest n_sites do not match this lattice.")

        self._reset_i_vn()
        self._reset_i_local()

        expected = self._expected_entropy_mask()
        filled = np.zeros_like(expected)

        for chunk in manifest["chunks"]:
            chunk_path = manifest_path.parent / chunk["output"]
            if not chunk_path.exists():
                raise FileNotFoundError(chunk_path)
            with np.load(chunk_path) as data:
                jobs = np.asarray(data["jobs"], dtype=int)
                values = np.asarray(data["values"], dtype=float)
            if np.any(filled[jobs[:, 0], jobs[:, 1], jobs[:, 2], jobs[:, 3], jobs[:, 4], jobs[:, 5]]):
                raise RuntimeError(f"Duplicate Slurm chunk results detected in {chunk_path}")
            self.apply_entropy_results(jobs, values)
            filled[jobs[:, 0], jobs[:, 1], jobs[:, 2], jobs[:, 3], jobs[:, 4], jobs[:, 5]] = True

        if not np.array_equal(filled, expected):
            raise RuntimeError("Missing or incomplete Slurm chunk results.")

        self.compute_local_information()
        return manifest

    def compute_local_information(self):
        self._reset_i_local()
        for lx, ly, lz, nx, ny, nz in self.entropy_jobs():
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
