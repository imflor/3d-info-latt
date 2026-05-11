import argparse
from pathlib import Path

import numpy as np

from .common import build_lattice, load_manifest, resolve_run_path


def main(default_manifest=None):
    parser = argparse.ArgumentParser(description="Assemble Slurm chunk outputs into a full 3D information lattice.")
    parser.add_argument("--manifest", default=default_manifest, help="Path to the manifest.json file.")
    args = parser.parse_args()

    if args.manifest is None:
        raise ValueError("Pass --manifest to choose which run to assemble.")

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)

    lat = build_lattice(manifest["lattice"], parallel="slurm", loader=False)
    lat.load_slurm_results(manifest_path)

    output_path = resolve_run_path(manifest_path, manifest["data"]["assembled"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        n_sites=lat.n_sites,
        i_vn=lat.i_vn,
        i_local=lat.i_local,
    )

    print(f"Assembled lattice to {output_path}")


if __name__ == "__main__":
    main()

