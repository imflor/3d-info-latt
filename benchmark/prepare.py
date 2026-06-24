import argparse
from pathlib import Path

import infolattice as il

from .common import (
    STAGE_SIZES,
    build_state_kwargs_from_values,
    resolve_n_chunks,
    run_name_for,
    state_slug,
    write_json,
    write_stage_script,
    write_submit_script,
)
from .profile_state import collect_state_profile


def prepare_run(args, stage, length, periodic):
    n_sites = (length, length, length)
    run_name = run_name_for(args.state_class, length, periodic)
    run_dir = Path(args.run_root) / run_name
    data_dir = run_dir / "data"
    manifest_path = run_dir / "manifest.json"
    submit_path = run_dir / "submit_array.slurm"
    correlation_relpath = Path("data") / "correlation.npy"
    correlation_path = run_dir / correlation_relpath

    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    state_kwargs = build_state_kwargs_from_values(
        args.state_class,
        n_sites,
        periodic=periodic,
        t=args.t,
        m=args.m,
        v=args.v,
        surface_mass=args.surface_mass,
    )
    state_cls = getattr(il, args.state_class)
    state = state_cls(**state_kwargs)
    state.save_correlation_matrix(correlation_path)

    profile_path = data_dir / "state_profile.json"
    if not args.skip_local_profile:
        write_json(
            profile_path,
            collect_state_profile(
                args.state_class,
                n_sites,
                periodic=periodic,
                t=args.t,
                m=args.m,
                v=args.v,
                surface_mass=args.surface_mass,
            ),
        )

    lat = il.InformationLattice(n_sites, parallel="slurm", loader=False, precompute_subsystems=False)
    n_chunks = resolve_n_chunks(lat, args.n_chunks, args.chunk_size, periodic=periodic)
    manifest = lat.write_slurm_manifest(
        manifest_path,
        run_name=run_name,
        state_name=args.state_class,
        state_kwargs=state_kwargs,
        state_path=correlation_relpath,
        state_periodic=periodic,
        n_chunks=n_chunks,
        shuffle_seed=args.shuffle_seed,
        output_name="lattice.npz",
    )
    manifest["benchmark"] = {
        "stage": stage,
        "state_class": args.state_class,
        "periodic": bool(periodic),
        "length": int(length),
        "n_sites": list(n_sites),
        "model_parameters": state_kwargs,
        "state_slug": state_slug(args.state_class),
        "state_profile": profile_path.relative_to(run_dir).as_posix() if profile_path.exists() else None,
        "worker_module": "benchmark.worker",
        "assembler_module": "benchmark.assemble",
        "rough_targets": [15, 20, 30],
    }
    write_json(manifest_path, manifest)
    write_submit_script(submit_path, manifest_path, manifest["n_chunks"], mem=args.mem)

    return {
        "stage": stage,
        "length": int(length),
        "periodic": bool(periodic),
        "run_name": run_name,
        "run_dir": run_dir.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "submit_script": submit_path.as_posix(),
        "n_jobs": int(manifest["n_jobs"]),
        "n_chunks": int(manifest["n_chunks"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare staged benchmark runs on top of the 3D Slurm workflow.")
    parser.add_argument("--stage", choices=sorted(STAGE_SIZES), help="Named benchmark stage to prepare.")
    parser.add_argument("--lengths", type=int, nargs="+", default=None, help="Explicit cubic system sizes L to benchmark.")
    parser.add_argument("--state-class", default="NodalLineGS", choices=["TightBindingGS", "NodalLineGS"])
    parser.add_argument("--run-root", default="benchmark/runs", help="Root directory for benchmark runs.")
    parser.add_argument("--generated-root", default="benchmark/generated", help="Root directory for generated helper scripts.")
    parser.add_argument("--n-chunks", type=int, default=128, help="Exact number of chunks per run.")
    parser.add_argument("--chunk-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--mem", default="16G", help="Slurm memory request for each benchmark worker.")
    parser.add_argument("--skip-local-profile", action="store_true", help="Skip local state-building probes during prepare.")
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=2.8)
    parser.add_argument("--v", type=float, default=1.0)
    parser.add_argument("--surface-mass", type=float, default=0.0)
    args = parser.parse_args()

    if args.stage is None and not args.lengths:
        raise ValueError("Pass either --stage or --lengths.")
    if args.stage is not None and args.lengths:
        raise ValueError("Pass either --stage or --lengths, not both.")

    if args.lengths:
        stage_name = "custom_" + "_".join(str(int(length)) for length in args.lengths)
        lengths = [int(length) for length in args.lengths]
    else:
        stage_name = args.stage
        lengths = STAGE_SIZES[args.stage]

    generated_dir = Path(args.generated_root) / stage_name
    generated_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for length in lengths:
        for periodic in (False, True):
            runs.append(prepare_run(args, stage_name, length, periodic))

    write_json(generated_dir / "runs.json", {"stage": stage_name, "runs": runs})
    write_stage_script(
        generated_dir / "submit_all.sh",
        [f"sbatch '{run['submit_script']}'" for run in runs],
    )
    write_stage_script(
        generated_dir / "assemble_all.sh",
        [f"python3 -m benchmark.assemble --manifest '{run['manifest_path']}'" for run in runs],
    )
    write_stage_script(
        generated_dir / "report.sh",
        [f"python3 -m benchmark.report --stage {args.stage}"] if args.stage is not None else ["python3 -m benchmark.report"],
    )

    print(f"Prepared {len(runs)} benchmark runs for {stage_name}.")
    print(f"Run index: {generated_dir / 'runs.json'}")
    print(f"Submit helper: {generated_dir / 'submit_all.sh'}")
    print(f"Assemble helper: {generated_dir / 'assemble_all.sh'}")


if __name__ == "__main__":
    main()
