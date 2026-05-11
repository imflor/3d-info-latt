import json
from pathlib import Path

import numpy as np
import infolattice as il


def load_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    return json.loads(manifest_path.read_text())


def resolve_run_path(manifest_path, relative_path):
    return Path(manifest_path).parent / relative_path


def build_state(manifest_path, state_spec):
    name = state_spec["name"]
    kwargs = dict(state_spec.get("kwargs", {}))
    state_path = state_spec.get("state_path")
    if state_path is not None:
        kwargs["correlation_path"] = str(resolve_run_path(manifest_path, state_path))
    cls = getattr(il, name, None)
    if cls is None:
        raise ValueError(f"Unknown state class '{name}' in manifest.")
    return cls(**kwargs)


def build_lattice(lattice_spec, *, parallel="none", loader=False):
    kwargs = dict(lattice_spec.get("kwargs", {}))
    kwargs["parallel"] = parallel
    kwargs["loader"] = loader
    kwargs["precompute_subsystems"] = False
    lat = il.InformationLattice(**kwargs)
    lat.batch_size = int(lattice_spec.get("batch_size", lat.batch_size))
    return lat


def load_chunk_jobs(manifest_path, manifest, chunk):
    jobs_path = resolve_run_path(manifest_path, manifest["data"]["jobs"])
    jobs = np.load(jobs_path)
    return jobs[int(chunk["start"]):int(chunk["stop"])]


def write_submit_script(submit_path, manifest_path, n_chunks, mem="4G"):
    """Overwrite the shared Slurm array script for the most recently prepared run."""
    submit_path = Path(submit_path)
    manifest_path = Path(manifest_path)
    array_max = max(int(n_chunks) - 1, 0)
    mem = str(mem)

    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=infolattice-chunk",
        f"#SBATCH --array=0-{array_max}",
        "#SBATCH --time=02:00:00",
        "#SBATCH --cpus-per-task=1",
        f"#SBATCH --mem={mem}",
        "#SBATCH --output=cluster/logs/%x_%A_%a.out",
        "",
        "# This file is overwritten by cluster.prepare_state or cluster.prepare_tight_binding.",
        "# Submit directly with:",
        "#   sbatch cluster/submit_array.slurm",
        "# Or override the manifest explicitly with:",
        "#   sbatch cluster/submit_array.slurm <manifest_path>",
        "",
        "set -euo pipefail",
        "",
        'cd "$SLURM_SUBMIT_DIR"',
        "mkdir -p cluster/logs",
        f'MANIFEST="${{1:-{manifest_path.as_posix()}}}"',
        'python -m cluster.worker --manifest "$MANIFEST" --chunk-id "${SLURM_ARRAY_TASK_ID}"',
        "",
    ]

    submit_path.write_text("\n".join(lines))
