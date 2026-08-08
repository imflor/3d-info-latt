import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPORT_PATH = Path("benchmark/reports/measured_summary.json")
FIGURE_PATH = Path("figures/benchmark_cumulative_loglog.png")
TARGET_LENGTHS = (20.0, 30.0)
FIT_DEGREE = 2
PHASES = [
    ("state_total_seconds", "state build"),
    ("sum_manifest_load_seconds", "worker manifest load"),
    ("sum_load_jobs_seconds", "worker jobs load"),
    ("sum_build_state_seconds", "worker state build"),
    ("sum_build_lattice_seconds", "worker lattice build"),
    ("sum_entropy_total_seconds", "worker entropy"),
    ("sum_write_output_seconds", "worker write"),
    ("assemble_seconds", "assemble"),
]


def format_hms(seconds):
    total = int(round(float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def fit_open_time(x, total):
    log_total = np.log(np.maximum(total, 1e-12))
    design = np.column_stack([np.ones_like(x), np.log(x), x])
    coeffs, _, _, _ = np.linalg.lstsq(design, log_total, rcond=None)

    def predict(target):
        features = np.array([1.0, np.log(target), target], dtype=float)
        return float(np.exp(features @ coeffs))

    return predict


def fit_periodic_time(log_x, total):
    log_total = np.log10(np.maximum(total, 1e-12))
    degree = min(FIT_DEGREE, len(log_x) - 1)
    fit = np.poly1d(np.polyfit(log_x, log_total, deg=degree))

    def predict(target):
        return float(10 ** fit(np.log10(target)))

    return fit, degree, predict


with REPORT_PATH.open() as f:
    records = json.load(f)["measured"]

groups = {}
for record in records:
    groups.setdefault(bool(record["periodic"]), []).append(record)

fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4), squeeze=False)
axes = axes[0]

title_parts = []
for ax, (periodic, rows) in zip(axes, sorted(groups.items())):
    rows = sorted(rows, key=lambda row: row["length"])
    x = np.array([row["length"] for row in rows], dtype=float)
    log_x = np.log10(x)

    labels = []
    series = []
    for key, label in PHASES:
        values = np.array([row.get(key, 0.0) for row in rows], dtype=float)
        if np.any(values > 0):
            labels.append(label)
            series.append(values)

    worker_parts = [
        np.array([row.get(key, 0.0) for row in rows], dtype=float)
        for key, _ in PHASES
        if key.startswith("sum_")
    ]
    worker_known = np.sum(worker_parts, axis=0) if worker_parts else np.zeros_like(x)
    worker_total = np.array([row.get("sum_worker_seconds", 0.0) for row in rows], dtype=float)
    extra = np.maximum(worker_total - worker_known, 0.0)
    if np.any(extra > 0):
        labels.append("other worker overhead")
        series.append(extra)

    lower = np.zeros_like(x)
    for values, label in zip(series, labels):
        upper = lower + values
        safe_lower = np.where(lower > 0, np.log10(lower), 0.0)
        safe_upper = np.log10(np.maximum(upper, 1e-12))
        ax.fill_between(log_x, safe_lower, safe_upper, alpha=0.9, label=label)
        lower = upper

    total = lower.copy()
    fit_x = np.linspace(log_x.min(), np.log10(TARGET_LENGTHS[-1]), 400)
    fit_lengths = 10 ** fit_x

    if periodic:
        fit, degree, predict = fit_periodic_time(log_x, total)
        fit_y = fit(fit_x)
        fit_label = f"poly fit (deg {degree})"
    else:
        predict = fit_open_time(x, total)
        fit_y = np.log10(np.maximum([predict(length) for length in fit_lengths], 1e-12))
        fit_label = r"fit $L^p e^{cL}$"

    ax.plot(
        fit_x,
        fit_y,
        linestyle="--",
        linewidth=1.5,
        color="black",
        alpha=0.8,
        label=fit_label,
    )

    estimates = {}
    for target in TARGET_LENGTHS:
        estimate = predict(target)
        estimates[int(target)] = estimate
        ax.scatter(
            [np.log10(target)],
            [np.log10(max(estimate, 1e-12))],
            color="black",
            marker="x",
            s=60,
            linewidths=2,
            label=f"L={int(target)} estimate" if target == TARGET_LENGTHS[0] else None,
            zorder=5,
        )

    name = "periodic" if periodic else "open"
    title_parts.append(
        f"{name} L20 {format_hms(estimates[20])} L30 {format_hms(estimates[30])}"
    )

    ax.set_title(name)
    ax.set_xlabel("log10(system size)")
    ax.set_ylabel("log10(total computation time [s])")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

fig.suptitle("Estimated total compute time: " + " | ".join(title_parts))
fig.tight_layout(rect=(0, 0, 1, 0.94))
FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
plt.show()
