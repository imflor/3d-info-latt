import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import infolattice as il

from .common import build_state_kwargs_from_values, peak_rss_mb, peak_rss_raw


ENTROPY_STAGE_SPECS = (
    ("entropy_unit", lambda L: (0, 0, 0)),
    ("entropy_line_half_x", lambda L: (max(0, math.ceil(L / 2) - 1), 0, 0)),
    ("entropy_cube_half_edge", lambda L: 3 * (max(0, math.ceil(L / 2) - 1),)),
    ("entropy_slab_xy_half_z", lambda L: (L - 1, L - 1, max(0, math.ceil(L / 2) - 1))),
)


def stage_names():
    return [
        "state_total",
        "hamiltonian",
        "diagonalization",
        "correlations",
        *[name for name, _ in ENTROPY_STAGE_SPECS],
    ]


def _entropy_shape(stage_name, length):
    for name, shape_fn in ENTROPY_STAGE_SPECS:
        if name == stage_name:
            return shape_fn(int(length))
    raise ValueError(f"Unknown entropy stage '{stage_name}'.")


def _build_bare_state(state_class, n_sites, kwargs):
    n_sites = np.array(n_sites, dtype=int)
    if state_class == "TightBindingGS":
        state = il.TightBindingGS.__new__(il.TightBindingGS)
        state.tol_log = 1e-16
        state.n_sites = n_sites
        state.nx, state.ny, state.nz = map(int, n_sites)
        state.n = int(n_sites.prod())
        state.periodic = False
        state.t = float(kwargs["t"])
        return state
    if state_class == "NodalLineGS":
        state = il.NodalLineGS.__new__(il.NodalLineGS)
        state.tol_log = 1e-16
        state.n_sites = n_sites
        state.nx, state.ny, state.nz = map(int, n_sites)
        state.n = int(n_sites.prod())
        state.periodic = bool(kwargs["periodic"])
        state.mass = float(kwargs["m"])
        state.v_orbital = float(kwargs["v"])
        state.surface_mass = float(kwargs["surface_mass"])
        return state
    raise ValueError(f"Unsupported state class '{state_class}'.")


def _probe_stage(stage_name, state_class, n_sites, kwargs):
    start = time.perf_counter()

    if stage_name == "state_total":
        getattr(il, state_class)(**kwargs)
        payload = {}
    elif stage_name == "hamiltonian":
        state = _build_bare_state(state_class, n_sites, kwargs)
        h = state.hamiltonian()
        payload = {"shape": list(h.shape)}
    elif stage_name == "diagonalization":
        state = _build_bare_state(state_class, n_sites, kwargs)
        h = state.hamiltonian()
        e, _ = np.linalg.eigh(h)
        payload = {"n_eigenvalues": int(len(e))}
    elif stage_name == "correlations":
        state = _build_bare_state(state_class, n_sites, kwargs)
        state.h = state.hamiltonian()
        state.e, state.v = state.diagonalize_hamiltonian()
        chi = state.correlations()
        payload = {"shape": list(chi.shape)}
    elif stage_name.startswith("entropy_"):
        state = getattr(il, state_class)(**kwargs)
        lat = il.InformationLattice(n_sites, parallel="none", loader=False, precompute_subsystems=False)
        lx, ly, lz = _entropy_shape(stage_name, int(n_sites[0]))
        sites = lat._get_subsystem_sites((0, 0, 0), (lx, ly, lz))
        entropy = state.entanglement_entropy(sites)
        payload = {
            "shape": [int(lx + 1), int(ly + 1), int(lz + 1)],
            "volume": int((lx + 1) * (ly + 1) * (lz + 1)),
            "entropy": float(entropy),
        }
    else:
        raise ValueError(f"Unknown stage '{stage_name}'.")

    seconds = time.perf_counter() - start
    rss_raw = peak_rss_raw()
    return {
        "stage": stage_name,
        "seconds": float(seconds),
        "peak_rss_raw": int(rss_raw),
        "peak_rss_mb": float(peak_rss_mb(rss_raw)),
        **payload,
    }


def collect_state_profile(state_class, n_sites, *, periodic=False, t=1.0, m=2.8, v=1.0, surface_mass=0.0):
    kwargs = build_state_kwargs_from_values(
        state_class,
        n_sites,
        periodic=periodic,
        t=t,
        m=m,
        v=v,
        surface_mass=surface_mass,
    )
    n_sites = tuple(int(x) for x in n_sites)
    stages = []
    for stage_name in stage_names():
        cmd = [
            sys.executable,
            "-m",
            "benchmark.profile_state",
            "--probe",
            "--stage",
            stage_name,
            "--state-class",
            state_class,
            "--n-sites",
            *(str(x) for x in n_sites),
            "--t",
            str(t),
            "--m",
            str(m),
            "--v",
            str(v),
            "--surface-mass",
            str(surface_mass),
        ]
        if periodic:
            cmd.append("--periodic")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        stages.append(json.loads(result.stdout))

    return {
        "state_class": state_class,
        "n_sites": list(n_sites),
        "periodic": bool(periodic),
        "state_kwargs": kwargs,
        "stages": stages,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile expensive free-fermion state-building stages.")
    parser.add_argument("--probe", action="store_true", help="Run one isolated profiling probe and print JSON.")
    parser.add_argument("--stage", choices=stage_names(), help="Profiling stage name.")
    parser.add_argument("--state-class", default="NodalLineGS", choices=["TightBindingGS", "NodalLineGS"])
    parser.add_argument("--n-sites", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=2.8)
    parser.add_argument("--v", type=float, default=1.0)
    parser.add_argument("--surface-mass", type=float, default=0.0)
    parser.add_argument("--output", default=None, help="Optional JSON output path for the full profile.")
    args = parser.parse_args()

    if args.probe:
        kwargs = build_state_kwargs_from_values(
            args.state_class,
            args.n_sites,
            periodic=args.periodic,
            t=args.t,
            m=args.m,
            v=args.v,
            surface_mass=args.surface_mass,
        )
        payload = _probe_stage(args.stage, args.state_class, tuple(args.n_sites), kwargs)
        sys.stdout.write(json.dumps(payload))
        return

    profile = collect_state_profile(
        args.state_class,
        tuple(args.n_sites),
        periodic=args.periodic,
        t=args.t,
        m=args.m,
        v=args.v,
        surface_mass=args.surface_mass,
    )
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(profile, indent=2) + "\n")
    else:
        print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()

