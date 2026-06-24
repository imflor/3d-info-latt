import argparse
import os
import time
from pathlib import Path

import numpy as np

from cluster.common import build_lattice, build_state, load_chunk_jobs, load_manifest, resolve_run_path

from .common import peak_rss_bytes, peak_rss_mb, peak_rss_raw


def main():
    worker_start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Benchmark one Slurm chunk of 3D free-fermion entropy jobs.")
    parser.add_argument("--manifest", required=True, help="Path to the manifest.json file.")
    parser.add_argument("--chunk-id", type=int, default=None, help="Chunk id. Defaults to SLURM_ARRAY_TASK_ID when available.")
    args = parser.parse_args()

    if args.chunk_id is None:
        if "SLURM_ARRAY_TASK_ID" not in os.environ:
            raise ValueError("Pass --chunk-id or set SLURM_ARRAY_TASK_ID.")
        chunk_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    else:
        chunk_id = int(args.chunk_id)

    manifest_load_start = time.perf_counter()
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    manifest_load_seconds = time.perf_counter() - manifest_load_start

    if chunk_id < 0 or chunk_id >= len(manifest["chunks"]):
        raise IndexError("chunk_id is outside the manifest range.")

    chunk = manifest["chunks"][chunk_id]

    load_jobs_start = time.perf_counter()
    jobs = load_chunk_jobs(manifest_path, manifest, chunk)
    load_jobs_seconds = time.perf_counter() - load_jobs_start

    subsystem_shapes = np.asarray(
        [(int(lx) + 1, int(ly) + 1, int(lz) + 1) for lx, ly, lz, _, _, _ in jobs],
        dtype=int,
    )
    subsystem_volumes = np.prod(subsystem_shapes, axis=1) if len(subsystem_shapes) else np.empty(0, dtype=int)

    print(f"chunk_id={chunk_id}")
    print(f"n_jobs={len(jobs)}")
    print(f"subsystem_shapes={subsystem_shapes.tolist()}")
    print(f"subsystem_volumes={subsystem_volumes.tolist()}")

    build_state_start = time.perf_counter()
    state = build_state(manifest_path, manifest["state"])
    build_state_seconds = time.perf_counter() - build_state_start

    build_lattice_start = time.perf_counter()
    lat = build_lattice(manifest["lattice"], parallel="none", loader=False)
    build_lattice_seconds = time.perf_counter() - build_lattice_start

    entropy_start = time.perf_counter()
    values = np.empty(len(jobs), dtype=float)
    entropy_seconds_per_job = np.empty(len(jobs), dtype=float)
    for idx, job in enumerate(jobs):
        lx, ly, lz, nx, ny, nz = map(int, job)
        t0 = time.perf_counter()
        sites = lat._subsystem_sites(lx, ly, lz, nx, ny, nz)
        values[idx] = (lx + 1) * (ly + 1) * (lz + 1) - state.entanglement_entropy(sites)
        entropy_seconds_per_job[idx] = time.perf_counter() - t0
    entropy_total_seconds = time.perf_counter() - entropy_start

    output_path = resolve_run_path(manifest_path, chunk["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f"{output_path.stem}.tmp.npz"

    write_output_start = time.perf_counter()
    worker_seconds = time.perf_counter() - worker_start
    rss_raw = peak_rss_raw()
    rss_bytes = peak_rss_bytes(rss_raw)
    rss_mb = peak_rss_mb(rss_raw)
    rss_gb = rss_bytes / (1024 ** 3)

    np.savez_compressed(
        tmp_path,
        chunk_id=int(chunk_id),
        jobs=np.asarray(jobs, dtype=int),
        values=values,
        subsystem_shapes=subsystem_shapes,
        subsystem_volumes=np.asarray(subsystem_volumes, dtype=int),
        manifest_load_seconds=float(manifest_load_seconds),
        load_jobs_seconds=float(load_jobs_seconds),
        build_state_seconds=float(build_state_seconds),
        build_lattice_seconds=float(build_lattice_seconds),
        entropy_total_seconds=float(entropy_total_seconds),
        entropy_seconds_per_job=entropy_seconds_per_job,
        worker_seconds=float(worker_seconds),
        peak_rss_raw=int(rss_raw),
        peak_rss_bytes=int(rss_bytes),
        peak_rss_mb=float(rss_mb),
    )
    tmp_path.replace(output_path)
    write_output_seconds = time.perf_counter() - write_output_start

    print(f"Wrote benchmark chunk {chunk_id} to {output_path}")
    print(f"manifest_load_seconds={manifest_load_seconds:.6f}")
    print(f"load_jobs_seconds={load_jobs_seconds:.6f}")
    print(f"build_state_seconds={build_state_seconds:.6f}")
    print(f"build_lattice_seconds={build_lattice_seconds:.6f}")
    print(f"entropy_total_seconds={entropy_total_seconds:.6f}")
    print(f"write_output_seconds={write_output_seconds:.6f}")
    print(f"worker_seconds={worker_seconds:.6f}")
    print(f"peak_rss_raw={int(rss_raw)}")
    print(f"peak_rss_mb={rss_mb:.3f}")
    print(f"peak_rss_gb={rss_gb:.3f}")


if __name__ == "__main__":
    main()

