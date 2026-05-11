# Cluster workflow

Run all commands below from the repository root.

This cluster workflow is for the existing 3D free-fermion tight-binding calculation. It parallelizes the subsystem entropy jobs over Slurm array workers and then assembles the final information-lattice arrays from the chunk outputs.

The workflow is:

1. Prepare a run
2. Inspect the generated run directory and manifest
3. Submit the Slurm array
4. Let each worker compute one chunk
5. Assemble the final lattice
6. Optionally sync the run directory back locally

## What the workflow computes

For a 3D lattice with dimensions `(Nx, Ny, Nz)`, the code enumerates all cuboid subsystems and computes their entropies using the existing `TightBindingGS.entanglement_entropy(...)` route.

The cluster workflow does not change the physics. It only changes how the entropy jobs are distributed:

* the prepare step writes the job list and chunk layout
* workers reconstruct the same free-fermion state from the manifest
* each worker computes only its assigned cuboid jobs
* assembly fills the full `i_vn` and `i_local` arrays

## Step 1: prepare a run

Use:

```bash
python -m cluster.prepare_tight_binding
```

By default this prepares a modest example with:

* `n_sites = (3, 3, 2)`
* `t = 1.0`

and writes the run into:

```text
cluster/runs/tight_binding_3x3x2/
```

### Important prepare parameters

The prepare script accepts the following user-facing parameters:

* `--n-sites NX NY NZ`
  Meaning: the 3D lattice dimensions.
  Example: `--n-sites 4 4 3`

* `--t FLOAT`
  Meaning: nearest-neighbor hopping amplitude in the tight-binding Hamiltonian.
  Example: `--t 1.0`

* `--n-chunks INT`
  Meaning: the exact number of Slurm chunks to create.
  Each array worker handles one chunk.
  Example: `--n-chunks 100`

* `--mem STRING`
  Meaning: memory request written into `cluster/submit_array.slurm`.
  Example: `--mem 16G`

* `--shuffle-seed INT`
  Meaning: seed used before weighted greedy balancing, so equal-weight jobs are distributed reproducibly.
  Example: `--shuffle-seed 7`

* `--run-dir PATH`
  Meaning: optional explicit output directory for the run.
  If omitted, the run directory is chosen automatically.

### Automatic run-directory naming

If `--run-dir` is not given, the prepare step names the run directory as:

```text
cluster/runs/tight_binding_{Nx}x{Ny}x{Nz}
```

For example:

* `cluster/runs/tight_binding_3x3x2`
* `cluster/runs/tight_binding_4x4x3`

### How chunking works

The prepare step does not create equal-size chunks by job count.

Instead it:

1. enumerates all cuboid entropy jobs
2. assigns each job a heuristic weight
3. creates exactly `n_chunks` chunks
4. uses weighted greedy assignment to balance heavy jobs across workers

The balancing weight is:

```text
2**m
```

where

```text
m = min(n_subsystem, N - n_subsystem)
```

and `n_subsystem = (lx + 1) * (ly + 1) * (lz + 1)`.

This is only a scheduling heuristic. The actual physics calculation is unchanged.

## Step 2: inspect the generated run directory

After preparing a run, inspect the output directory before submitting.

For example:

```bash
python -m cluster.prepare_tight_binding --n-sites 4 4 3 --n-chunks 100 --t 1.0 --mem 16G
```

This will create:

```text
cluster/runs/tight_binding_4x4x3/
```

with files such as:

```text
cluster/runs/tight_binding_4x4x3/manifest.json
cluster/runs/tight_binding_4x4x3/data/jobs.npy
cluster/runs/tight_binding_4x4x3/data/chunks/
```

You can inspect the manifest with:

```bash
python -m json.tool cluster/runs/tight_binding_4x4x3/manifest.json
```

The manifest records:

* the state class and its parameters
* the lattice size
* the chunk layout
* the output paths
* the number of jobs and number of chunks

## Step 3: submit the Slurm array

The prepare script automatically rewrites:

```text
cluster/submit_array.slurm
```

for the most recently prepared run.

This means it updates:

* the `#SBATCH --array=...` range
* the default manifest path used by the worker
* the `#SBATCH --mem=...` setting

So after preparing a run, you can submit directly with:

```bash
sbatch cluster/submit_array.slurm
```

If you want to override the manifest explicitly, you can still do:

```bash
sbatch cluster/submit_array.slurm cluster/runs/tight_binding_4x4x3/manifest.json
```

## Step 4: what each worker does

Each Slurm array task runs:

```bash
python -m cluster.worker --manifest <manifest_path> --chunk-id <chunk_id>
```

The worker then:

1. reads the manifest
2. loads its assigned chunk from `data/jobs.npy`
3. reconstructs the `TightBindingGS` state from manifest parameters
4. computes the entropy values for only its assigned cuboids
5. writes one chunk output file

Each chunk file is written as:

```text
cluster/runs/<run_name>/data/chunks/chunk_XXXXX.npz
```

Each worker also prints useful metadata to stdout, including:

* `chunk_id`
* `n_jobs`
* `subsystem_shapes`
* `subsystem_volumes`
* `worker_seconds`

So the Slurm log tells you both what the worker handled and how long it took.

## Step 5: assemble the final result

After all chunk files exist, assemble them into the full lattice arrays:

```bash
python -m cluster.assemble --manifest cluster/runs/tight_binding_4x4x3/manifest.json
```

This step:

1. reads the manifest
2. loads all chunk outputs
3. fills `i_vn`
4. computes `i_local`
5. writes the assembled result as a compressed `.npz`

For the example above, the assembled output is:

```text
cluster/runs/tight_binding_4x4x3/data/tight_binding_lattice.npz
```

That file contains:

* `n_sites`
* `i_vn`
* `i_local`

## Full worked example

Here is a full start-to-finish example you can copy.

### 1. Prepare

```bash
python -m cluster.prepare_tight_binding --n-sites 4 4 3 --n-chunks 100 --t 1.0 --mem 16G
```

### 2. Inspect the manifest

```bash
python -m json.tool cluster/runs/tight_binding_4x4x3/manifest.json
```

### 3. Submit the array

```bash
sbatch cluster/submit_array.slurm
```

### 4. Assemble after completion

```bash
python -m cluster.assemble --manifest cluster/runs/tight_binding_4x4x3/manifest.json
```

### 5. Result location

```text
cluster/runs/tight_binding_4x4x3/data/tight_binding_lattice.npz
```

## Sync results back locally

If the run happens on a remote cluster checkout, you can sync the run directory back to your local repository afterward.

For example, from your local machine:

```bash
rsync -av "user@cluster-host:~/3d-info-latt/cluster/runs/" "cluster/runs/"
```

Adjust the remote path to match where you cloned the repository on the cluster.

## Troubleshooting

### Missing chunk outputs during assembly

If assembly fails because some chunk outputs are missing or incomplete:

* inspect `cluster/runs/<run_name>/manifest.json`
* check which chunk files exist under `cluster/runs/<run_name>/data/chunks/`
* inspect the corresponding Slurm logs in `cluster/logs/`
* rerun or resubmit the missing chunk ids if needed

The assembler expects one completed `.npz` file per chunk listed in the manifest.

### Wrong Slurm array range

If the array range in `cluster/submit_array.slurm` does not match the run you intended to submit, the most likely reason is that the file was overwritten by a different prepare step.

The fix is simple:

1. rerun the prepare command for the run you actually want
2. verify the rewritten `cluster/submit_array.slurm`
3. then submit again

### Where results are written

The main outputs are:

* manifest: `cluster/runs/<run_name>/manifest.json`
* job list: `cluster/runs/<run_name>/data/jobs.npy`
* chunk outputs: `cluster/runs/<run_name>/data/chunks/chunk_*.npz`
* assembled lattice: `cluster/runs/<run_name>/data/tight_binding_lattice.npz`

### Reassembling without rerunning workers

If all chunk files already exist, you do not need to rerun the workers.
You can just rerun:

```bash
python -m cluster.assemble --manifest cluster/runs/<run_name>/manifest.json
```
