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

rows = []
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        for k in metrics:
            row[k] = float(row[k])
        rows.append(row)

by = defaultdict(list)
for row in rows:
    by[row["group"]].append(row)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
for ax, metric in zip(axes, metrics):
    means = []
    mins = []
    maxs = []
    for g in order:
        vals = [r[metric] for r in by[g]]
        means.append(statistics.mean(vals))
        mins.append(min(vals))
        maxs.append(max(vals))
    x = np.arange(len(order))
    yerr = np.vstack([np.array(means) - np.array(mins), np.array(maxs) - np.array(means)])
    ax.errorbar(x, means, yerr=yerr, fmt='o', color='#4C78A8', ecolor='#4C78A8', elinewidth=2, capsize=4, markersize=6)
    ax.set_xticks(x, labels)
    ax.set_title(f"{metric} range", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)
    for xi, mv in zip(x, means):
        ax.text(xi, mv, f"{mv:.3f}", ha='center', va='bottom', fontsize=8)
fig.suptitle("SocialGood grouped min-mean-max ranges", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.93])
for ext in ["png", "svg"]:
    fig.savefig(os.path.join(OUT_DIR, f"socialgood_range_plot.{ext}"), dpi=300, bbox_inches="tight")
