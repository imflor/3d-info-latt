import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPORT_PATH = Path("benchmark/reports/measured_summary.json")
FIGURE_PATH = Path("figures/benchmark_memory_loglog.png")
TARGET_LENGTHS = (20.0, 30.0)
MEMORY_KEY = "max_worker_peak_rss_mb"
FIT_DEGREE = 2


with REPORT_PATH.open() as f:
    records = json.load(f)["measured"]

records = [row for row in records if float(row.get(MEMORY_KEY, 0.0)) > 0.0]

groups = {}
for record in records:
    groups.setdefault(bool(record["periodic"]), []).append(record)

fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4), squeeze=False)
axes = axes[0]

title_parts = []
for ax, (periodic, rows) in zip(axes, sorted(groups.items())):
    rows = sorted(rows, key=lambda row: row["length"])
    x = np.array([row["length"] for row in rows], dtype=float)
    y = np.array([row[MEMORY_KEY] for row in rows], dtype=float)
    log_x = np.log10(x)
    log_y = np.log10(y)

    ax.plot(log_x, log_y, marker="o", linewidth=2, label="measured")

    degree = min(FIT_DEGREE, len(x) - 1)
    fit = np.poly1d(np.polyfit(log_x, log_y, deg=degree))
    fit_x = np.linspace(log_x.min(), np.log10(TARGET_LENGTHS[-1]), 400)
    ax.plot(fit_x, fit(fit_x), linestyle="--", linewidth=1.5, color="black", alpha=0.8, label=f"poly fit (deg {degree})")

    estimates = {}
    for target in TARGET_LENGTHS:
        estimate = float(10 ** fit(np.log10(target)))
        estimates[int(target)] = estimate
        ax.scatter(
            [np.log10(target)],
            [np.log10(max(estimate, 1e-12))],
            color="black",
            marker="x",
            s=70,
            linewidths=2,
            label=f"L={int(target)} estimate" if target == TARGET_LENGTHS[0] else None,
            zorder=5,
        )

    name = "periodic" if periodic else "open"
    title_parts.append(
        f"{name} L20 {estimates[20] / 1024.0:.2f} GB L30 {estimates[30] / 1024.0:.2f} GB"
    )

    ax.set_title(name)
    ax.set_xlabel("log10(system size)")
    ax.set_ylabel("log10(max worker peak RSS [MB])")
    ax.legend(frameon=False)

fig.suptitle("Estimated max worker memory: " + " | ".join(title_parts))
fig.tight_layout(rect=(0, 0, 1, 0.94))
FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
plt.show()
