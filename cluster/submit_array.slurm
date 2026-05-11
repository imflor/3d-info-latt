#!/bin/bash
#SBATCH --job-name=infolattice-chunk
#SBATCH --array=0-0
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=cluster/logs/%x_%A_%a.out

# This file is overwritten by cluster.prepare_tight_binding.
# Submit directly with:
#   sbatch cluster/submit_array.slurm
# Or override the manifest explicitly with:
#   sbatch cluster/submit_array.slurm <manifest_path>

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p cluster/logs
MANIFEST="${1:-cluster/runs/tight_binding_3x3x2/manifest.json}"
python -m cluster.worker --manifest "$MANIFEST" --chunk-id "${SLURM_ARRAY_TASK_ID}"

