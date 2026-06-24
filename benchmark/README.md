# Benchmark workflow

Run all commands below from the repository root.

This folder is a disposable benchmark layer on top of the existing 3D Slurm workflow. It does not introduce a new parallelization system. Instead it reuses:

* the existing free-fermion state classes
* the existing manifest and chunk layout
* the existing cuboid entropy calculation path
* the existing assembled `i_vn` / `i_local` data products

The benchmark code adds:

* local profiling of the main state-building stages
* a benchmark worker that records chunk-level timing and peak RSS
* a benchmark assembler that records full-run timing and peak RSS
* stage scripts for the requested size sweeps
* a reporting script that writes measured summaries and coarse extrapolations

## What is being measured

The benchmark workflow targets the meaningful expensive stages:

* Hamiltonian construction
* single-particle diagonalization
* correlation-matrix construction
* representative subsystem-entropy calls
* full worker chunk time
* assembly of the final information lattice

For chunk workers it also records:

* per-job entropy timings
* total entropy time inside the worker
* total worker time
* peak RSS of the worker process

## Benchmark stages

The staged size sets are:

* `stage1`: `L = 5, 6, 7`
* `stage2`: `L = 8, 9, 10, 11`

Each stage prepares both:

* `periodic=False`
* `periodic=True`

By default the benchmark targets `NodalLineGS`, which is the most relevant current 3D free-fermion case. You can switch to `TightBindingGS` with `--state-class TightBindingGS`.

## Step 1: prepare a stage

Quick validation stage:

```bash
python -m benchmark.prepare --stage stage1 --state-class NodalLineGS --n-chunks 128 --mem 16G
```

Larger stage:

```bash
python -m benchmark.prepare --stage stage2 --state-class NodalLineGS --n-chunks 128 --mem 16G
```

Explicit size list, for example `L = 4, 5, 6, 7, 8, 9`:

```bash
python -m benchmark.prepare --lengths 4 5 6 7 8 9 --state-class NodalLineGS --n-chunks 128 --mem 16G
```

This creates benchmark runs under:

```text
benchmark/runs/
```

and helper scripts under:

```text
benchmark/generated/stage1/
benchmark/generated/stage2/
```

Each run directory contains:

```text
benchmark/runs/<run_name>/
  manifest.json
  submit_array.slurm
  data/
    correlation.npy
    jobs.npy
    state_profile.json
    chunks/
```

The prepare step also writes:

* `benchmark/generated/<stage>/runs.json`
* `benchmark/generated/<stage>/submit_all.sh`
* `benchmark/generated/<stage>/assemble_all.sh`
* `benchmark/generated/<stage>/report.sh`

### Important prepare parameters

* `--stage stage1|stage2`
  Chooses the requested size set.

* `--lengths L1 L2 ...`
  Uses an explicit custom list of cubic system sizes instead of a named stage.

* `--state-class NodalLineGS|TightBindingGS`
  Chooses the free-fermion model to benchmark.

* `--n-chunks INT`
  Sets the exact number of Slurm chunks per run.

* `--mem STRING`
  Sets the worker memory written into each benchmark submit script.

* `--skip-local-profile`
  Skips the local state-building probes if you only want to prepare the cluster runs.

* `--m`, `--v`, `--surface-mass`, `--t`
  Pass through the model parameters of the chosen state class.

## Step 2: inspect the prepared runs

For example:

```bash
python -m json.tool benchmark/runs/bench_nodal_line_L5_periodic/manifest.json
python -m json.tool benchmark/runs/bench_nodal_line_L5_periodic/data/state_profile.json
```

The manifest stores:

* run metadata
* state class and model parameters
* periodic/open setting
* the chunk layout
* the output paths
* benchmark-specific metadata

## Step 3: submit the Slurm workers

Each run gets its own submit script:

```text
benchmark/runs/<run_name>/submit_array.slurm
```

The benchmark submit script explicitly activates the `info-latt` environment when `conda.sh` is available under `$HOME/miniconda3`, and otherwise falls back to the interpreter at:

```text
$HOME/miniconda3/envs/info-latt/bin/python
```

You can submit them individually:

```bash
sbatch benchmark/runs/bench_nodal_line_L5_periodic/submit_array.slurm
```

or use the generated stage helper:

```bash
bash benchmark/generated/stage1/submit_all.sh
```

These worker jobs use the same manifest-and-chunk execution model as the production cluster workflow, but call `benchmark.worker` so that extra timing and memory data are recorded.

## Step 4: assemble the finished runs

After the chunk files exist, assemble each run:

```bash
python -m benchmark.assemble --manifest benchmark/runs/bench_nodal_line_L5_periodic/manifest.json
```

or assemble the whole stage:

```bash
bash benchmark/generated/stage1/assemble_all.sh
```

This writes:

```text
benchmark/runs/<run_name>/data/lattice.npz
benchmark/runs/<run_name>/data/assemble_metrics.json
```

## Step 5: generate the summary and extrapolations

After you have measured runs, build the report:

```bash
python -m benchmark.report --stage stage1
```

or after both stages:

```bash
python -m benchmark.report --stage stage1 --stage stage2
```

This writes:

```text
benchmark/reports/measured_summary.json
benchmark/reports/measured_summary.csv
benchmark/reports/extrapolated_estimates.json
benchmark/reports/extrapolated_estimates.csv
```

The report keeps measured data and extrapolated data separate.

## Output contents

### `state_profile.json`

Local state-building probes, including:

* `state_total`
* `hamiltonian`
* `diagonalization`
* `correlations`
* representative entropy stages such as:
  * `entropy_unit`
  * `entropy_line_half_x`
  * `entropy_cube_half_edge`
  * `entropy_slab_xy_half_z`

Each stage stores time and peak RSS.

### worker chunk `.npz`

Each chunk file stores:

* `jobs`
* `values`
* `subsystem_shapes`
* `subsystem_volumes`
* `entropy_seconds_per_job`
* `entropy_total_seconds`
* `build_state_seconds`
* `build_lattice_seconds`
* `worker_seconds`
* `peak_rss_raw`
* `peak_rss_bytes`
* `peak_rss_mb`

### `assemble_metrics.json`

Full-run assembly metrics, including:

* `assemble_seconds`
* `load_results_seconds`
* `write_output_seconds`
* `sum_worker_seconds`
* `max_worker_seconds`
* `max_worker_peak_rss_mb`
* assembler peak RSS

## Extrapolation notes

The extrapolation script fits simple log-log trends from the measured data and reports coarse estimates for:

* `L = 15`
* `L = 20`
* `L = 30`

It also includes a dense one-body memory context based on the matrix dimension of the chosen state class:

* `TightBindingGS`: one orbital per site
* `NodalLineGS`: two orbitals per site

Those dense memory numbers are context only. They do not replace the measured timings or RSS values.
