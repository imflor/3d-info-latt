#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate info-latt
fi

sbatch 'benchmark/runs/bench_nodal_line_L4_open/submit_array.slurm'
sbatch 'benchmark/runs/bench_nodal_line_L4_periodic/submit_array.slurm'
sbatch 'benchmark/runs/bench_nodal_line_L5_open/submit_array.slurm'
sbatch 'benchmark/runs/bench_nodal_line_L5_periodic/submit_array.slurm'
sbatch 'benchmark/runs/bench_nodal_line_L6_open/submit_array.slurm'
sbatch 'benchmark/runs/bench_nodal_line_L6_periodic/submit_array.slurm'
sbatch 'benchmark/runs/bench_nodal_line_L7_open/submit_array.slurm'
sbatch 'benchmark/runs/bench_nodal_line_L7_periodic/submit_array.slurm'
sbatch 'benchmark/runs/bench_nodal_line_L8_open/submit_array.slurm'
sbatch 'benchmark/runs/bench_nodal_line_L8_periodic/submit_array.slurm'
sbatch 'benchmark/runs/bench_nodal_line_L9_open/submit_array.slurm'
sbatch 'benchmark/runs/bench_nodal_line_L9_periodic/submit_array.slurm'
