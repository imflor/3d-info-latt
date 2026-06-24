import argparse
import csv
from pathlib import Path

import numpy as np

from .common import TARGET_SIZES, dense_context_gb, flatten_profile, load_json, orbitals_per_site, write_json


def iter_manifests(run_root):
    for manifest_path in sorted(Path(run_root).glob("*/manifest.json")):
        manifest = load_json(manifest_path)
        benchmark = manifest.get("benchmark")
        if benchmark is None:
            continue
        yield manifest_path, manifest, benchmark


def load_run_record(manifest_path, manifest, benchmark):
    run_dir = manifest_path.parent
    length = int(benchmark["length"])
    state_class = benchmark["state_class"]
    record = {
        "run_name": manifest.get("run_name"),
        "stage": benchmark["stage"],
        "state_class": state_class,
        "periodic": bool(benchmark["periodic"]),
        "length": length,
        "n_jobs": int(manifest["n_jobs"]),
        "n_chunks": int(manifest["n_chunks"]),
        "orbitals_per_site": orbitals_per_site(state_class),
    }
    record.update(dense_context_gb(length, state_class))

    profile_relpath = benchmark.get("state_profile")
    if profile_relpath:
        profile_path = run_dir / profile_relpath
        if profile_path.exists():
            record.update(flatten_profile(load_json(profile_path)))

    assemble_path = run_dir / "data" / "assemble_metrics.json"
    if assemble_path.exists():
        record.update(load_json(assemble_path))

    chunk_peak = []
    chunk_worker = []
    chunk_entropy = []
    for chunk in manifest["chunks"]:
        chunk_path = run_dir / chunk["output"]
        if not chunk_path.exists():
            continue
        data = np.load(chunk_path)
        if "peak_rss_mb" in data:
            chunk_peak.append(float(data["peak_rss_mb"]))
        if "worker_seconds" in data:
            chunk_worker.append(float(data["worker_seconds"]))
        if "entropy_total_seconds" in data:
            chunk_entropy.append(float(data["entropy_total_seconds"]))
        data.close()

    if chunk_worker:
        record.setdefault("sum_worker_seconds", float(sum(chunk_worker)))
        record.setdefault("max_worker_seconds", float(max(chunk_worker)))
        record.setdefault("mean_worker_seconds", float(np.mean(chunk_worker)))
    if chunk_entropy:
        record.setdefault("sum_entropy_total_seconds", float(sum(chunk_entropy)))
    if chunk_peak:
        record.setdefault("max_worker_peak_rss_mb", float(max(chunk_peak)))

    if "state_total_seconds" in record and "max_worker_seconds" in record and "assemble_seconds" in record:
        record["estimated_wall_seconds"] = (
            float(record["state_total_seconds"])
            + float(record["max_worker_seconds"])
            + float(record["assemble_seconds"])
        )
    if "state_total_seconds" in record and "sum_worker_seconds" in record and "assemble_seconds" in record:
        record["estimated_cpu_seconds"] = (
            float(record["state_total_seconds"])
            + float(record["sum_worker_seconds"])
            + float(record["assemble_seconds"])
        )

    return record


def extrapolate(records, targets):
    grouped = {}
    numeric_blacklist = {"length", "n_jobs", "n_chunks", "sites", "matrix_dim", "cuboids", "periodic"}
    for record in records:
        group = (record["state_class"], bool(record["periodic"]))
        grouped.setdefault(group, []).append(record)

    estimates = []
    for (state_class, periodic), rows in grouped.items():
        metric_names = sorted({
            key
            for row in rows
            for key, value in row.items()
            if key not in numeric_blacklist and isinstance(value, (int, float)) and not isinstance(value, bool)
        })
        for metric in metric_names:
            xy = [
                (float(row["length"]), float(row[metric]))
                for row in rows
                if isinstance(row.get(metric), (int, float)) and float(row[metric]) > 0
            ]
            if len(xy) < 2:
                continue
            xs = np.log(np.asarray([x for x, _ in xy], dtype=float))
            ys = np.log(np.asarray([y for _, y in xy], dtype=float))
            slope, intercept = np.polyfit(xs, ys, deg=1)
            for target in targets:
                value = float(np.exp(intercept + slope * np.log(float(target))))
                context = dense_context_gb(target, state_class)
                estimates.append({
                    "state_class": state_class,
                    "periodic": periodic,
                    "metric": metric,
                    "target_length": int(target),
                    "estimated_value": value,
                    "fit_exponent": float(slope),
                    "n_samples": int(len(xy)),
                    **context,
                })
    return estimates


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark measurements and coarse extrapolations.")
    parser.add_argument("--run-root", default="benchmark/runs")
    parser.add_argument("--report-root", default="benchmark/reports")
    parser.add_argument("--stage", action="append", choices=["stage1", "stage2"], help="Optional stage filter.")
    args = parser.parse_args()

    records = []
    for manifest_path, manifest, benchmark in iter_manifests(args.run_root):
        if args.stage and benchmark["stage"] not in args.stage:
            continue
        records.append(load_run_record(manifest_path, manifest, benchmark))

    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    summary_json = report_root / "measured_summary.json"
    summary_csv = report_root / "measured_summary.csv"
    estimates_json = report_root / "extrapolated_estimates.json"
    estimates_csv = report_root / "extrapolated_estimates.csv"

    write_json(summary_json, {"measured": records})
    write_csv(summary_csv, records)

    estimates = extrapolate(records, TARGET_SIZES)
    write_json(estimates_json, {"targets": TARGET_SIZES, "estimates": estimates})
    write_csv(estimates_csv, estimates)

    print(f"Wrote measured summary to {summary_json}")
    print(f"Wrote measured CSV to {summary_csv}")
    print(f"Wrote extrapolated estimates to {estimates_json}")
    print(f"Wrote extrapolated CSV to {estimates_csv}")


if __name__ == "__main__":
    main()
