import argparse
import math
from pathlib import Path

import infolattice as il

from .common import write_submit_script


DEFAULT_CHUNK_SIZE = 200
STATE_SPECS = {
    "TightBindingGS": {
        "slug": "tight_binding",
    },
    "NodalLineGS": {
        "slug": "nodal_line",
    },
}


def resolve_n_chunks(lat, n_chunks, chunk_size, periodic=False):
    if n_chunks is not None and chunk_size is not None:
        raise ValueError("Pass either --n-chunks or --chunk-size, not both.")
    if n_chunks is not None:
        n_chunks = int(n_chunks)
        if n_chunks <= 0:
            raise ValueError("--n-chunks must be positive.")
        return n_chunks

    chunk_size = DEFAULT_CHUNK_SIZE if chunk_size is None else int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    n_jobs = len(lat.entropy_jobs(periodic=periodic))
    return max(1, math.ceil(n_jobs / chunk_size))


def state_slug(state_class):
    if state_class in STATE_SPECS:
        return STATE_SPECS[state_class]["slug"]
    raise ValueError(f"Unsupported state class '{state_class}'.")


def build_state_kwargs(args, state_class, n_sites):
    state_kwargs = {"n_sites": list(n_sites)}
    if state_class == "TightBindingGS":
        state_kwargs["t"] = args.t
        return state_kwargs
    if state_class == "NodalLineGS":
        state_kwargs["periodic"] = args.periodic
        state_kwargs["m"] = args.m
        state_kwargs["v"] = args.v
        state_kwargs["surface_mass"] = args.surface_mass
        return state_kwargs
    raise ValueError(f"Unsupported state class '{state_class}'.")


def main(default_state_class="TightBindingGS", allow_state_class=True):
    parser = argparse.ArgumentParser(description="Prepare a Slurm manifest for a 3D free-fermion information-lattice run.")
    parser.add_argument("--run-dir", default=None, help="Parent directory that will hold the run folder. Defaults to cluster/runs/.")
    parser.add_argument("--run-name", default=None, help="Simulation ID and run-folder name.")
    parser.add_argument("--n-sites", type=int, nargs=3, default=(3, 3, 2), metavar=("NX", "NY", "NZ"), help="3D lattice dimensions.")
    if allow_state_class:
        parser.add_argument(
            "--state-class",
            default=default_state_class,
            choices=sorted(STATE_SPECS),
            help="Free-fermion state class to prepare.",
        )
    parser.add_argument("--n-chunks", type=int, default=None, help="Exact number of Slurm chunks to create.")
    parser.add_argument("--chunk-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shuffle-seed", type=int, default=0, help="Seed used to shuffle jobs before chunking.")
    parser.add_argument("--mem", default="4G", help="Memory requested per Slurm worker.")
    parser.add_argument("--t", type=float, default=1.0, help="Nearest-neighbor hopping for TightBindingGS.")
    parser.add_argument("--periodic", action="store_true", help="Use periodic boundaries for NodalLineGS.")
    parser.add_argument("--m", type=float, default=2.8, help="Mass parameter for NodalLineGS.")
    parser.add_argument("--v", type=float, default=1.0, help="Orbital-mixing parameter for NodalLineGS.")
    parser.add_argument("--surface-mass", type=float, default=0.0, help="Boundary mixing mass for NodalLineGS.")
    args = parser.parse_args()

    state_class = default_state_class if not allow_state_class else args.state_class
    n_sites = tuple(int(x) for x in args.n_sites)
    slug = state_slug(state_class)

    run_name = args.run_name
    if run_name is None:
        run_name = f"{slug}_{n_sites[0]}x{n_sites[1]}x{n_sites[2]}"

    run_name_path = Path(run_name)
    if run_name_path.name != run_name or run_name in {".", ".."}:
        raise ValueError("--run-name must be a single folder name, not a path.")

    base_run_dir = Path("cluster") / "runs" if args.run_dir is None else Path(args.run_dir)
    run_dir = base_run_dir / run_name
    data_dir = run_dir / "data"
    manifest_path = run_dir / "manifest.json"
    submit_path = Path("cluster") / "submit_array.slurm"
    correlation_relpath = Path("data") / "correlation.npy"
    correlation_path = run_dir / correlation_relpath

    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    state_kwargs = build_state_kwargs(args, state_class, n_sites)
    state_cls = getattr(il, state_class, None)
    if state_cls is None:
        raise ValueError(f"Unknown state class '{state_class}'.")
    state = state_cls(**state_kwargs)
    if not hasattr(state, "save_correlation_matrix"):
        raise TypeError(f"State class '{state_class}' does not support saved correlation matrices.")
    state.save_correlation_matrix(correlation_path)
    state_periodic = bool(getattr(state, "periodic", False))

    lat = il.InformationLattice(n_sites, parallel="slurm", loader=False)
    n_chunks = resolve_n_chunks(lat, args.n_chunks, args.chunk_size, periodic=state_periodic)
    manifest = lat.write_slurm_manifest(
        manifest_path,
        run_name=run_name,
        state_name=state_class,
        state_kwargs=state_kwargs,
        state_path=correlation_relpath,
        state_periodic=state_periodic,
        n_chunks=n_chunks,
        shuffle_seed=args.shuffle_seed,
        output_name="lattice.npz",
    )
    write_submit_script(submit_path, manifest_path, manifest["n_chunks"], mem=args.mem)

    print(f"run_name={run_name}")
    print(f"state_class={state_class}")
    print(f"Wrote correlation matrix to {correlation_path}")
    print(f"Wrote Slurm manifest to {manifest_path}")
    print(f"Updated shared submit script at {submit_path}")
    print(f"n_jobs={manifest['n_jobs']}")
    print(f"n_chunks={manifest['n_chunks']}")
    print(f"shuffle_seed={manifest['shuffle_seed']}")
    print(f"state_periodic={state_periodic}")
    print(f"worker_mem={args.mem}")
    print("You can now submit directly with: sbatch cluster/submit_array.slurm")


if __name__ == "__main__":
    main()
