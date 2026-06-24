import json
import resource
import sys
from pathlib import Path

import numpy as np

from cluster.prepare_state import build_state_kwargs, resolve_n_chunks, state_slug


STAGE_SIZES = {
    "stage1": [5, 6, 7],
    "stage2": [8, 9, 10, 11],
}
TARGET_SIZES = [15, 20, 30]


def peak_rss_raw():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def peak_rss_bytes(raw_value):
    raw_value = int(raw_value)
    if sys.platform == "darwin":
        return raw_value
    return raw_value * 1024


def peak_rss_mb(raw_value):
    return peak_rss_bytes(raw_value) / (1024 ** 2)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_json(path):
    return json.loads(Path(path).read_text())


def orbitals_per_site(state_class):
    return {
        "TightBindingGS": 1,
        "NodalLineGS": 2,
    }.get(state_class, 1)


def dense_context_gb(length, state_class):
    length = int(length)
    sites = length ** 3
    dim = orbitals_per_site(state_class) * sites
    raw_gb = (dim ** 2) * 8 / 1e9
    return {
        "sites": sites,
        "matrix_dim": dim,
        "raw_dense_gb": raw_gb,
        "rough_dense_lower_gb": 3.0 * raw_gb,
        "rough_dense_upper_gb": 6.0 * raw_gb,
        "cuboids": ((length * (length + 1)) // 2) ** 3,
    }


def run_name_for(state_class, length, periodic):
    boundary = "periodic" if periodic else "open"
    return f"bench_{state_slug(state_class)}_L{int(length)}_{boundary}"


def build_state_kwargs_from_values(state_class, n_sites, *, periodic=False, t=1.0, m=2.8, v=1.0, surface_mass=0.0):
    class Args:
        pass

    args = Args()
    args.t = t
    args.periodic = periodic
    args.m = m
    args.v = v
    args.surface_mass = surface_mass
    return build_state_kwargs(args, state_class, tuple(int(x) for x in n_sites))


def write_submit_script(submit_path, manifest_path, n_chunks, *, mem="4G", worker_module="benchmark.worker"):
    submit_path = Path(submit_path)
    manifest_path = Path(manifest_path)
    array_max = max(int(n_chunks) - 1, 0)

    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=infolattice-bench",
        f"#SBATCH --array=0-{array_max}",
        "#SBATCH --time=04:00:00",
        "#SBATCH --cpus-per-task=1",
        f"#SBATCH --mem={mem}",
        "#SBATCH --output=benchmark/logs/%x_%A_%a.out",
        "",
        "set -euo pipefail",
        "",
        'cd "$SLURM_SUBMIT_DIR"',
        "mkdir -p benchmark/logs",
        'CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"',
        'if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then',
        '  source "$CONDA_BASE/etc/profile.d/conda.sh"',
        '  conda activate info-latt',
        '  PYTHON_BIN="${PYTHON_BIN:-python}"',
        "else",
        '  PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/info-latt/bin/python}"',
        "fi",
        'if [ ! -x "$(command -v "$PYTHON_BIN" 2>/dev/null || true)" ] && [ ! -x "$PYTHON_BIN" ]; then',
        '  echo "Could not find info-latt Python interpreter." >&2',
        "  exit 1",
        "fi",
        f'MANIFEST="${{1:-{manifest_path.as_posix()}}}"',
        f'"$PYTHON_BIN" -m {worker_module} --manifest "$MANIFEST" --chunk-id "${{SLURM_ARRAY_TASK_ID}}"',
        "",
    ]
    submit_path.parent.mkdir(parents=True, exist_ok=True)
    submit_path.write_text("\n".join(lines))


def write_stage_script(script_path, lines):
    script_path = Path(script_path)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    prelude = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        "REPO_ROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/../../..\" && pwd)\"",
        "cd \"$REPO_ROOT\"",
        "CONDA_BASE=\"${CONDA_BASE:-$HOME/miniconda3}\"",
        "if [ -f \"$CONDA_BASE/etc/profile.d/conda.sh\" ]; then",
        "  source \"$CONDA_BASE/etc/profile.d/conda.sh\"",
        "  conda activate info-latt",
        "fi",
        "",
    ]
    script_path.write_text("\n".join(prelude + list(lines)) + "\n")


def flatten_profile(profile):
    flat = {}
    for entry in profile.get("stages", []):
        stage = entry["stage"]
        flat[f"{stage}_seconds"] = entry.get("seconds")
        flat[f"{stage}_peak_rss_mb"] = entry.get("peak_rss_mb")
        if "volume" in entry:
            flat[f"{stage}_volume"] = entry["volume"]
    return flat
