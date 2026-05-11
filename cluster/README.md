# Cluster workflow

Run commands from the repository root.

## Tight-binding prepare step

Prepare a 3D free-fermion run with:

```bash
python -m cluster.prepare_tight_binding
```

This writes a run directory such as `cluster/runs/tight_binding_3x3x2/` containing:

* `manifest.json`
* `data/jobs.npy`
* `data/chunks/chunk_*.npz`

The prepare step also overwrites the shared file `cluster/submit_array.slurm` so its array range and default manifest point to the most recently prepared run.
Jobs are assigned into exactly `n_chunks` chunks using weighted greedy balancing.
The balancing weight is `2**m`, with `m = min(n_subsystem, N - n_subsystem)` and `n_subsystem = (lx + 1) * (ly + 1) * (lz + 1)`.
By default the prepare script infers a reasonable `n_chunks` from the old chunk-size heuristic, but the primary interface is now `--n-chunks`.
The optional `--shuffle-seed` still shuffles jobs before weighting so equal-weight ties are distributed reproducibly.
You can set the worker memory at prepare time, for example:

```bash
python -m cluster.prepare_tight_binding --n-sites 4 4 3 --n-chunks 100 --mem 16G
```

## Submit worker jobs

After running the prepare step, you can submit directly with:

```bash
sbatch cluster/submit_array.slurm
```

If needed, you can still override the manifest explicitly:

```bash
sbatch cluster/submit_array.slurm cluster/runs/tight_binding_3x3x2/manifest.json
```

The prepare step rewrites `cluster/submit_array.slurm` so its `#SBATCH --array` line already matches `0 .. n_chunks - 1` for the most recently prepared run, and its `#SBATCH --mem` line matches the `--mem` value from that prepare command.

## Worker tasks

Each array task runs:

```bash
python -m cluster.worker --manifest <manifest_path> --chunk-id <chunk_id>
```

The worker reconstructs the `TightBindingGS` state from the manifest, computes only its assigned cuboid jobs, and writes one chunk file.
The worker also prints `subsystem_shapes=[...]`, `subsystem_volumes=[...]`, and `worker_seconds=...` to stdout so the Slurm log records what that task handled and how long it took.

## Assemble the result

After all chunk outputs exist:

```bash
python -m cluster.assemble --manifest <manifest_path>
```

For example:

```bash
python -m cluster.assemble --manifest cluster/runs/tight_binding_3x3x2/manifest.json
```
