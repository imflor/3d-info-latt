## Code for the information lattice

This repository contains:

* `infolattice/`: the core importable package with the 3D lattice, physics, parallel, and plotting code.
* `01_singlets.py`, `02_diamond_fermi_surface.py`: lightweight numbered experiment scripts that use `import infolattice as il`.

## Setup

Requires Python >= 3.9.

All scripts assume the working directory is the repository root.

### Install dependencies (recommended)

Create and activate a virtual environment (or use an existing environment such as Conda), then install this repository:

```bash
python -m venv .venv

# macOS/Linux
. .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
```

`python -m pip install -e .` installs the required dependencies and makes the `infolattice` package importable.

## Running the code

### Run an experiment

Run an experiment script from the repository root, for example:

```bash
python 01_singlets.py
```
