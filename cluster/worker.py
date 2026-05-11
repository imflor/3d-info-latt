import argparse
import os
import time
from pathlib import Path

import numpy as np

from .common import build_lattice, build_state, load_chunk_jobs, load_manifest, resolve_run_path


def main():
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="Compute one Slurm chunk of cuboid entropies.")
    parser.add_argument("--manifest", required=True, help="Path to the manifest.json file.")
    parser.add_argument("--chunk-id", type=int, default=None, help="Chunk id. Defaults to SLURM_ARRAY_TASK_ID when available.")
    args = parser.parse_args()

    if args.chunk_id is None:
        if "SLURM_ARRAY_TASK_ID" not in os.environ:
            raise ValueError("Pass --chunk-id or set SLURM_ARRAY_TASK_ID.")
        chunk_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    else:
        chunk_id = int(args.chunk_id)

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)

    if chunk_id < 0 or chunk_id >= len(manifest["chunks"]):
        raise IndexError("chunk_id is outside the manifest range.")

    chunk = manifest["chunks"][chunk_id]
    jobs = load_chunk_jobs(manifest_path, manifest, chunk)
    subsystem_shapes = [(int(lx) + 1, int(ly) + 1, int(lz) + 1) for lx, ly, lz, _, _, _ in jobs]
    subsystem_volumes = [sx * sy * sz for sx, sy, sz in subsystem_shapes]

    print(f"chunk_id={chunk_id}")
    print(f"n_jobs={len(jobs)}")
    print(f"subsystem_shapes={subsystem_shapes}")
    print(f"subsystem_volumes={subsystem_volumes}")

    state = build_state(manifest_path, manifest["state"])
    lat = build_lattice(manifest["lattice"], parallel="none", loader=False)
    values = lat.entropy_values_for_jobs(state, jobs)
    output_path = resolve_run_path(manifest_path, chunk["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f"{output_path.stem}.tmp.npz"

    np.savez_compressed(
        tmp_path,
        chunk_id=int(chunk_id),
        jobs=np.asarray(jobs, dtype=int),
        values=values,
    )
    tmp_path.replace(output_path)

    print(f"Wrote chunk {chunk_id} to {output_path}")
    print(f"worker_seconds={time.perf_counter() - start_time:.6f}")


if __name__ == "__main__":
    main()
