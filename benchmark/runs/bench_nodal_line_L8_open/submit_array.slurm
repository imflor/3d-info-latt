#!/bin/bash
#SBATCH --job-name=infolattice-bench
#SBATCH --array=0-127
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=benchmark/logs/%x_%A_%a.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p benchmark/logs
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate info-latt
  PYTHON_BIN="${PYTHON_BIN:-python}"
else
  PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/info-latt/bin/python}"
fi
if [ ! -x "$(command -v "$PYTHON_BIN" 2>/dev/null || true)" ] && [ ! -x "$PYTHON_BIN" ]; then
  echo "Could not find info-latt Python interpreter." >&2
  exit 1
fi
MANIFEST="${1:-benchmark/runs/bench_nodal_line_L8_open/manifest.json}"
"$PYTHON_BIN" -m benchmark.worker --manifest "$MANIFEST" --chunk-id "${SLURM_ARRAY_TASK_ID}"
