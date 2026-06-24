#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate info-latt
fi

python3 -m benchmark.assemble --manifest 'benchmark/runs/bench_nodal_line_L4_open/manifest.json'
python3 -m benchmark.assemble --manifest 'benchmark/runs/bench_nodal_line_L4_periodic/manifest.json'
python3 -m benchmark.assemble --manifest 'benchmark/runs/bench_nodal_line_L5_open/manifest.json'
python3 -m benchmark.assemble --manifest 'benchmark/runs/bench_nodal_line_L5_periodic/manifest.json'
python3 -m benchmark.assemble --manifest 'benchmark/runs/bench_nodal_line_L6_open/manifest.json'
python3 -m benchmark.assemble --manifest 'benchmark/runs/bench_nodal_line_L6_periodic/manifest.json'
python3 -m benchmark.assemble --manifest 'benchmark/runs/bench_nodal_line_L7_open/manifest.json'
python3 -m benchmark.assemble --manifest 'benchmark/runs/bench_nodal_line_L7_periodic/manifest.json'
python3 -m benchmark.assemble --manifest 'benchmark/runs/bench_nodal_line_L8_open/manifest.json'
python3 -m benchmark.assemble --manifest 'benchmark/runs/bench_nodal_line_L8_periodic/manifest.json'
python3 -m benchmark.assemble --manifest 'benchmark/runs/bench_nodal_line_L9_open/manifest.json'
python3 -m benchmark.assemble --manifest 'benchmark/runs/bench_nodal_line_L9_periodic/manifest.json'
