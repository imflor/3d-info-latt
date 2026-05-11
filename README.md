## 3D Information Lattice

This repository contains code for computing the 3D information lattice on cuboid subsystems of a finite lattice.

The repository has two main parts:

* `infolattice/`: the importable package with the 3D lattice logic, physics models, parallel helpers, and plotting tools
* `cluster/`: the Slurm-oriented workflow for preparing, distributing, and assembling free-fermion entropy calculations

It also includes two lightweight numbered experiment scripts:

* `01_singlets.py`
* `02_diamond_fermi_surface.py`

## Setup

Requires Python >= 3.9.

All commands below assume the working directory is the repository root.

### Install dependencies

Create and activate a virtual environment, or use an existing environment such as Conda, then install this repository:

```bash
python -m venv .venv

# macOS/Linux
. .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
```

Installing with `-e` makes `infolattice` importable from the repository root and from scripts inside the repo.

## Local usage

### Run a local experiment

For example:

```bash
python 01_singlets.py
```

or

```bash
python 02_diamond_fermi_surface.py
```

These scripts are meant as simple local examples and do not use the cluster workflow.

## Cluster workflow

The cluster workflow is for the free-fermion state classes in this repository. It does not add any DMRG or interacting-state machinery.

The workflow is:

1. Prepare a run and write a manifest
2. Submit a Slurm array job
3. Let each worker compute one entropy chunk
4. Assemble all chunk outputs into the final lattice arrays
5. Optionally sync the run directory back to your local machine

The operational guide lives in:

* [`cluster/README.md`](/Users/matthiasflor/Documents/1.%20PhD/3.%20Information%20lattice/3d-info-latt/cluster/README.md)

### Quick cluster example

Prepare a `4 x 4 x 3` run for `TightBindingGS` split into `100` chunks:

```bash
python -m cluster.prepare_state --state-class TightBindingGS --n-sites 4 4 3 --n-chunks 100 --t 1.0 --mem 16G
```

This writes a run directory such as:

```text
cluster/runs/tight_binding_4x4x3/
```

with a manifest at:

```text
cluster/runs/tight_binding_4x4x3/manifest.json
```

The prepare step also rewrites:

```text
cluster/submit_array.slurm
```

so that it is immediately ready for this run, including:

* the correct `#SBATCH --array=...` range
* the correct default manifest path
* the requested worker memory

You can then submit:

```bash
sbatch cluster/submit_array.slurm
```

After all chunks finish, assemble:

```bash
python -m cluster.assemble --manifest cluster/runs/tight_binding_4x4x3/manifest.json
```

The assembled data is written to:

```text
cluster/runs/tight_binding_4x4x3/data/lattice.npz
```

For a nodal-line run, you would instead use for example:

```bash
python -m cluster.prepare_state --state-class NodalLineGS --n-sites 4 4 3 --n-chunks 100 --m 2.8 --v 1.0 --surface-mass 0.0 --mem 16G
```

If you want to choose the run folder name explicitly, you can now do for example:

```bash
python -m cluster.prepare_state --state-class NodalLineGS --n-sites 4 4 3 --n-chunks 100 --m 2.8 --v 1.0 --surface-mass 0.0 --run-name nodal_line_surface_scan --mem 16G
```

Here `--run-name` is the simulation ID. It determines the run folder name, and the workflow then uses fixed filenames inside that folder such as `manifest.json` and `data/lattice.npz`.

For backward compatibility, the old tight-binding-only entry point still works:

```bash
python -m cluster.prepare_tight_binding --n-sites 4 4 3 --n-chunks 100 --t 1.0 --mem 16G
```

## Where to look next

If you want the full step-by-step cluster instructions, including:

* what the manifest contains
* what each CLI flag means
* what each worker does
* how to inspect run directories
* how to sync results back locally
* what to do when chunks are missing

read:

* [`cluster/README.md`](/Users/matthiasflor/Documents/1.%20PhD/3.%20Information%20lattice/3d-info-latt/cluster/README.md)
