"""Plots for the nested-CV ablation (run after src.ablation_nested)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "ablation"
FIG = ROOT / "Thesis" / "figures"
ORDER = ["baseline", "ga_only", "pso_only", "sequential", "cooperative"]
LABEL = {"baseline": "Baseline", "ga_only": "GA only", "pso_only": "PSO only",
         "sequential": "Sequential\nGA→PSO", "cooperative": "Cooperative\nGA–PSO"}

df = pd.DataFrame(json.loads((RES / "ablation_raw.json").read_text()))

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (met, name) in zip(axes, [("acc", "Accuracy"), ("f1", "F1 score"), ("mcc", "MCC")]):
    data = [df[df.variant == v][met].to_numpy() for v in ORDER]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor("#2a6fb0" if ORDER[i] == "cooperative" else "#c9d6e3")
    for i, d in enumerate(data):
        ax.scatter([i + 1] * len(d), d, s=10, color="#333333", alpha=0.6, zorder=3)
    ax.set_xticks(range(1, len(ORDER) + 1))
    ax.set_xticklabels([LABEL[v] for v in ORDER], fontsize=8)
    ax.set_title(name)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Outer-fold test scores, nested CV (5 folds × 3 seeds)")
fig.tight_layout()
fig.savefig(FIG / "ablation_boxplot.png", dpi=200)

feat = pd.read_csv(RES / "coop_feature_frequency.csv", index_col=0).iloc[:, 0]
feat = (feat / len(df[df.variant == "cooperative"])).sort_values()
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.barh(feat.index, feat.values, color="#2a6fb0")
ax.set_xlabel("Fraction of the 15 outer folds in which the feature was selected")
ax.set_xlim(0, 1)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "ablation_feature_stability.png", dpi=200)
print("saved")
