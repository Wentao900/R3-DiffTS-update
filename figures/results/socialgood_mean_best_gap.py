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

focus = ["plain", "notext", "scorehack"]
metrics = ["CRPS", "MSE", "MAE"]
colors = {"mean": "#4C78A8", "best": "#E45756"}

rows = []
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        for k in metrics:
            row[k] = float(row[k])
        rows.append(row)

by = defaultdict(list)
for row in rows:
    by[row["group"]].append(row)

summary = {}
for g in focus:
    summary[g] = {}
    for m in metrics:
        vals = [r[m] for r in by[g]]
        summary[g][f"{m}_mean"] = statistics.mean(vals)
        summary[g][f"{m}_best"] = min(vals)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
width = 0.35
x = np.arange(len(focus))
for ax, metric in zip(axes, metrics):
    mean_vals = [summary[g][f"{metric}_mean"] for g in focus]
    best_vals = [summary[g][f"{metric}_best"] for g in focus]
    b1 = ax.bar(x - width/2, mean_vals, width, label="group mean", color=colors["mean"], edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width/2, best_vals, width, label="best run", color=colors["best"], edgecolor="black", linewidth=0.6)
    ax.set_xticks(x, focus)
    ax.set_title(metric, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for bars, vals in [(b1, mean_vals), (b2, best_vals)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=8)
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Mean vs best-run comparison on SocialGood", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.93])
for ext in ["png", "svg"]:
    fig.savefig(os.path.join(OUT_DIR, f"socialgood_mean_best_gap.{ext}"), dpi=300, bbox_inches="tight")
