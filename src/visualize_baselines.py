"""Generate all baseline visualizations: confusion matrices, ROC curves,
metric bar charts, CV box plots, and a summary heatmap.

Run from the repository root:
    python -m src.visualize_baselines
"""

from pathlib import Path
import json
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.preprocessing import load_and_preprocess

# Paths
DATA_RAW_DIR = Path("data") / "raw"
RESULTS_DIR = Path("results") / "baseline"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
MODEL_COLORS = {
    "SVM (baseline)": "#4C72B0",
    "Random Forest (baseline)": "#55A868",
    "ANN / MLP (baseline)": "#C44E52",
}
METRICS = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]


def build_models():
    svm_base = SVC(kernel="rbf", random_state=42)
    return {
        "SVM (baseline)": CalibratedClassifierCV(svm_base, ensemble=False),
        "Random Forest (baseline)": RandomForestClassifier(n_estimators=100, random_state=42),
        "ANN / MLP (baseline)": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42),
    }


def plot_confusion_matrices(dataset_name, x_test, y_test, models_fitted):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Confusion Matrices — {dataset_name.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    for ax, (model_name, model) in zip(axes, models_fitted.items()):
        ConfusionMatrixDisplay.from_estimator(
            model, x_test, y_test,
            display_labels=["No Disease", "Disease"],
            cmap="Blues", ax=ax, colorbar=False
        )
        ax.set_title(model_name.replace(" (baseline)", ""), fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"{dataset_name}_confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_roc_curves(dataset_name, x_test, y_test, models_fitted):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
    for model_name, model in models_fitted.items():
        probas = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probas)
        roc_auc = auc(fpr, tpr)
        short = model_name.replace(" (baseline)", "")
        ax.plot(fpr, tpr, lw=2, color=MODEL_COLORS[model_name], label=f"{short} (AUC={roc_auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {dataset_name.replace('_', ' ').title()}", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = PLOTS_DIR / f"{dataset_name}_roc_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_pr_curves(dataset_name, x_test, y_test, models_fitted):
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, model in models_fitted.items():
        probas = model.predict_proba(x_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, probas)
        pr_auc = auc(recall, precision)
        short = model_name.replace(" (baseline)", "")
        ax.plot(recall, precision, lw=2, color=MODEL_COLORS[model_name], label=f"{short} (AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves — {dataset_name.replace('_', ' ').title()}", fontweight="bold")
    ax.legend(loc="lower left")
    plt.tight_layout()
    out = PLOTS_DIR / f"{dataset_name}_pr_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_metric_bars(results, dataset_name):
    model_names = list(results.keys())
    short_names = [n.replace(" (baseline)", "") for n in model_names]
    x = np.arange(len(METRICS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model_name, short) in enumerate(zip(model_names, short_names)):
        values = [results[model_name][m] for m in METRICS]
        bars = ax.bar(x + i * width, values, width, label=short,
                      color=MODEL_COLORS[model_name], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(METRIC_LABELS, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(f"Baseline Model Comparison — {dataset_name.replace('_', ' ').title()}", fontweight="bold", fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = PLOTS_DIR / f"{dataset_name}_metric_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_cv_boxplot(dataset_name, x_train, y_train, models_fitted):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_scores = {}
    for model_name, model in models_fitted.items():
        scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")
        all_scores[model_name.replace(" (baseline)", "")] = scores
    fig, ax = plt.subplots(figsize=(8, 5))
    data = list(all_scores.values())
    labels = list(all_scores.keys())
    colors = [MODEL_COLORS[k + " (baseline)"] for k in labels]
    bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("5-Fold CV Accuracy")
    ax.set_title(f"CV Accuracy Distribution — {dataset_name.replace('_', ' ').title()}", fontweight="bold")
    ax.set_ylim(0.5, 1.05)
    plt.tight_layout()
    out = PLOTS_DIR / f"{dataset_name}_cv_boxplot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_summary_heatmap(all_results):
    rows = []
    row_labels = []
    for dataset_name, results in all_results.items():
        short_ds = "Cleveland" if "cleveland" in dataset_name.lower() else "Statlog"
        for model_name, metrics in results.items():
            short_model = model_name.replace(" (baseline)", "")
            row_labels.append(f"{short_ds} — {short_model}")
            rows.append([metrics[m] for m in METRICS])
    data = np.array(rows)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=METRIC_LABELS, yticklabels=row_labels,
                linewidths=0.5, linecolor="white", ax=ax, vmin=0.7, vmax=1.0)
    ax.set_title("Baseline Results Summary Heatmap", fontweight="bold", fontsize=13, pad=12)
    ax.set_xticklabels(METRIC_LABELS, rotation=30, ha="right")
    plt.tight_layout()
    out = PLOTS_DIR / "summary_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_cv_comparison(all_results):
    datasets = list(all_results.keys())
    short_ds = {d: ("Cleveland" if "cleveland" in d.lower() else "Statlog") for d in datasets}
    model_names = list(next(iter(all_results.values())).keys())
    short_models = [m.replace(" (baseline)", "") for m in model_names]
    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = ["#2196F3", "#FF9800"]
    for i, dataset in enumerate(datasets):
        means = [all_results[dataset][m]["cv_mean_accuracy"] for m in model_names]
        stds = [all_results[dataset][m]["cv_std_accuracy"] for m in model_names]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=5,
                      label=short_ds[dataset], alpha=0.82, edgecolor="white",
                      color=palette[i], error_kw=dict(elinewidth=1.5, ecolor="gray"))
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.018,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(short_models, fontsize=11)
    ax.set_ylim(0.6, 1.08)
    ax.set_ylabel("5-Fold CV Accuracy (Mean +/- Std)")
    ax.set_title("CV Accuracy Comparison Across Datasets", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out = PLOTS_DIR / "cv_accuracy_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    warnings.filterwarnings("ignore")
    csv_paths = sorted(DATA_RAW_DIR.glob("*.csv"))
    all_results = {}

    for csv_path in csv_paths:
        json_path = RESULTS_DIR / f"{csv_path.stem}_baseline_results.json"
        if not json_path.exists():
            print(f"Skipping {csv_path.stem}: no JSON results found.")
            continue
        try:
            prepared = load_and_preprocess(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path.stem}: {e}")
            continue

        x_train, x_test, y_train, y_test, feature_names, _ = prepared

        with json_path.open() as f:
            results = json.load(f)

        all_results[csv_path.stem] = results
        print(f"\n{'='*60}\nGenerating plots for: {csv_path.stem}")

        models = build_models()
        models_fitted = {}
        for name, model in models.items():
            model.fit(x_train, y_train)
            models_fitted[name] = model

        plot_confusion_matrices(csv_path.stem, x_test, y_test, models_fitted)
        plot_roc_curves(csv_path.stem, x_test, y_test, models_fitted)
        plot_pr_curves(csv_path.stem, x_test, y_test, models_fitted)
        plot_metric_bars(results, csv_path.stem)
        plot_cv_boxplot(csv_path.stem, x_train, y_train, models_fitted)

    if all_results:
        print(f"\n{'='*60}\nGenerating cross-dataset summary plots")
        plot_summary_heatmap(all_results)
        plot_cv_comparison(all_results)

    print(f"\nAll plots saved to: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
