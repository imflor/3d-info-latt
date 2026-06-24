import json

import matplotlib.pyplot as plt
import numpy as np


REPORT_PATH = "benchmark/reports/measured_summary.json"
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


with open(REPORT_PATH) as f:
    records = json.load(f)["measured"]

groups = {}
for record in records:
    groups.setdefault(bool(record["periodic"]), []).append(record)

fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4), squeeze=False)
axes = axes[0]

for ax, (periodic, rows) in zip(axes, sorted(groups.items())):
    rows = sorted(rows, key=lambda row: row["length"])
    x = np.array([row["length"] for row in rows], dtype=float)

    labels = []
    series = []
    for key, label in PHASES:
        values = np.array([row.get(key, 0.0) for row in rows], dtype=float)
        if np.any(values > 0):
            labels.append(label)
            series.append(values)

    known = np.sum(series, axis=0) if series else np.zeros_like(x)
    worker_total = np.array([row.get("sum_worker_seconds", 0.0) for row in rows], dtype=float)
    extra = np.maximum(worker_total - np.sum(
        [np.array([row.get(key, 0.0) for row in rows], dtype=float) for key, _ in PHASES if key.startswith("sum_")],
        axis=0,
    ) if any(key.startswith("sum_") for key, _ in PHASES) else worker_total, 0.0)
    if np.any(extra > 0):
        labels.append("other worker overhead")
        series.append(extra)

    ax.stackplot(x, np.array(series), labels=labels, alpha=0.9)
    ax.set_title("periodic" if periodic else "open")
    ax.set_xlabel("system size")
    ax.set_ylabel("total computation time [s]")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

fig.tight_layout()
plt.show()
