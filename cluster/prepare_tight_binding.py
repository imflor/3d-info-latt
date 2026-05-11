import argparse
import math
from pathlib import Path

import infolattice as il

from .common import write_submit_script


DEFAULT_CHUNK_SIZE = 200


def resolve_n_chunks(lat, n_chunks, chunk_size):
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
    n_jobs = len(lat.entropy_jobs())
    return max(1, math.ceil(n_jobs / chunk_size))


def main():
    parser = argparse.ArgumentParser(description="Prepare a Slurm manifest for a 3D tight-binding information-lattice run.")
    parser.add_argument("--run-dir", default=None, help="Directory that will hold the manifest and chunk data.")
    parser.add_argument("--n-sites", type=int, nargs=3, default=(3, 3, 2), metavar=("NX", "NY", "NZ"), help="3D lattice dimensions.")
    parser.add_argument("--n-chunks", type=int, default=None, help="Exact number of Slurm chunks to create.")
    parser.add_argument("--chunk-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shuffle-seed", type=int, default=0, help="Seed used to shuffle jobs before chunking.")
    parser.add_argument("--mem", default="4G", help="Memory requested per Slurm worker.")
    parser.add_argument("--t", type=float, default=1.0, help="Nearest-neighbor hopping.")
    args = parser.parse_args()

    n_sites = tuple(int(x) for x in args.n_sites)
    if args.run_dir is None:
        run_dir = Path("cluster") / "runs" / f"tight_binding_{n_sites[0]}x{n_sites[1]}x{n_sites[2]}"
    else:
        run_dir = Path(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    submit_path = Path("cluster") / "submit_array.slurm"

    lat = il.InformationLattice(n_sites, parallel="slurm", loader=False)
    n_chunks = resolve_n_chunks(lat, args.n_chunks, args.chunk_size)
    manifest = lat.write_slurm_manifest(
        manifest_path,
        state_name="TightBindingGS",
        state_kwargs={
            "n_sites": list(n_sites),
            "t": args.t,
        },
        n_chunks=n_chunks,
        shuffle_seed=args.shuffle_seed,
        output_name="tight_binding_lattice.npz",
    )
    write_submit_script(submit_path, manifest_path, manifest["n_chunks"], mem=args.mem)

    print(f"Wrote Slurm manifest to {manifest_path}")
    print(f"Updated shared submit script at {submit_path}")
    print(f"n_jobs={manifest['n_jobs']}")
    print(f"n_chunks={manifest['n_chunks']}")
    print(f"shuffle_seed={manifest['shuffle_seed']}")
    print(f"worker_mem={args.mem}")
    print("You can now submit directly with: sbatch cluster/submit_array.slurm")


if __name__ == "__main__":
    main()

