import csv
import os
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/root/autodl-tmp/R3-DiffTS-update"
CSV_PATH = os.path.join(ROOT, "tables", "socialgood_grouped_results.csv")
OUT_DIR = os.path.join(ROOT, "figures", "results")
os.makedirs(OUT_DIR, exist_ok=True)

order = ["plain", "notext", "scorehack", "backbone_coarse", "conservative"]
labels = ["plain", "notext", "scorehack", "backbone\ncoarse", "conservative"]
metrics = ["CRPS", "MSE", "MAE"]
colors = {
    "CRPS": "#4C78A8",
    "MSE": "#F58518",
    "MAE": "#54A24B",
}

rows = []
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        for k in metrics:
            row[k] = float(row[k])
        rows.append(row)

by = defaultdict(list)
for row in rows:
    by[row["group"]].append(row)

summary = {
    g: {m: statistics.mean(r[m] for r in by[g]) for m in metrics}
    for g in order if g in by
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
for ax, metric in zip(axes, metrics):
    vals = [summary[g][metric] for g in order]
    bars = ax.bar(labels, vals, color=colors[metric], alpha=0.88, edgecolor="black", linewidth=0.6)
    ax.set_title(f"SocialGood {metric} mean", fontsize=12)
    ax.set_ylabel(metric)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=8)

fig.suptitle("SocialGood grouped mean comparison", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ["png", "svg"]:
    fig.savefig(os.path.join(OUT_DIR, f"socialgood_group_mean_bars.{ext}"), dpi=300, bbox_inches="tight")
