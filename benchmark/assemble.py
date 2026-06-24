import argparse
import time
from pathlib import Path

import numpy as np

from cluster.common import build_lattice, load_manifest, resolve_run_path

from .common import peak_rss_bytes, peak_rss_mb, peak_rss_raw, write_json


def main():
    parser = argparse.ArgumentParser(description="Assemble benchmark chunk outputs into a full 3D information lattice.")
    parser.add_argument("--manifest", required=True, help="Path to the manifest.json file.")
    args = parser.parse_args()

    total_start = time.perf_counter()
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)

    build_lattice_start = time.perf_counter()
    lat = build_lattice(manifest["lattice"], parallel="slurm", loader=False)
    build_lattice_seconds = time.perf_counter() - build_lattice_start

    load_results_start = time.perf_counter()
    lat.load_slurm_results(manifest_path)
    load_results_seconds = time.perf_counter() - load_results_start

    worker_seconds = []
    entropy_seconds = []
    peak_memories = []
    for chunk in manifest["chunks"]:
        chunk_path = resolve_run_path(manifest_path, chunk["output"])
        with np.load(chunk_path) as data:
            if "worker_seconds" in data:
                worker_seconds.append(float(data["worker_seconds"]))
            if "entropy_total_seconds" in data:
                entropy_seconds.append(float(data["entropy_total_seconds"]))
            if "peak_rss_mb" in data:
                peak_memories.append(float(data["peak_rss_mb"]))

    output_path = resolve_run_path(manifest_path, manifest["data"]["assembled"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_output_start = time.perf_counter()
    np.savez_compressed(
        output_path,
        n_sites=lat.n_sites,
        i_vn=lat.i_vn,
        i_local=lat.i_local,
    )
    write_output_seconds = time.perf_counter() - write_output_start

    total_seconds = time.perf_counter() - total_start
    rss_raw = peak_rss_raw()
    metrics = {
        "run_name": manifest.get("run_name"),
        "state_class": manifest["state"]["name"],
        "periodic": bool(manifest["state"].get("periodic", False)),
        "n_sites": manifest["lattice"]["kwargs"]["n_sites"],
        "n_jobs": int(manifest["n_jobs"]),
        "n_chunks": int(manifest["n_chunks"]),
        "build_lattice_seconds": float(build_lattice_seconds),
        "load_results_seconds": float(load_results_seconds),
        "write_output_seconds": float(write_output_seconds),
        "assemble_seconds": float(total_seconds),
        "sum_worker_seconds": float(sum(worker_seconds)) if worker_seconds else None,
        "max_worker_seconds": float(max(worker_seconds)) if worker_seconds else None,
        "mean_worker_seconds": float(np.mean(worker_seconds)) if worker_seconds else None,
        "sum_entropy_total_seconds": float(sum(entropy_seconds)) if entropy_seconds else None,
        "max_worker_peak_rss_mb": float(max(peak_memories)) if peak_memories else None,
        "peak_rss_raw": int(rss_raw),
        "peak_rss_bytes": int(peak_rss_bytes(rss_raw)),
        "peak_rss_mb": float(peak_rss_mb(rss_raw)),
        "assembled_output": output_path.as_posix(),
    }
    write_json(output_path.parent / "assemble_metrics.json", metrics)

    print(f"Assembled lattice to {output_path}")
    print(f"assemble_seconds={total_seconds:.6f}")
    print(f"peak_rss_mb={metrics['peak_rss_mb']:.3f}")
    if metrics["max_worker_peak_rss_mb"] is not None:
        print(f"max_worker_peak_rss_mb={metrics['max_worker_peak_rss_mb']:.3f}")


if __name__ == "__main__":
    main()

