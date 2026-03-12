## Code for the information lattice

This repository contains:

* `demo/`: a minimal demo generating the information lattice for a given many-body state in 2 dimensions.

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

`python -m pip install -e .` installs the required dependencies and makes the shared `utils/` module importable.

## Running the code

### Run the minimal demo

Run the demo script from the repository root, for example:

```bash
python demo/main.py
```
