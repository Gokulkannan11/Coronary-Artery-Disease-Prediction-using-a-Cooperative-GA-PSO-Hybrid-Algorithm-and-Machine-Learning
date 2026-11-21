"""Full evaluation report generator for Cooperative GA-PSO results.

Reads the best features + hyperparameters from a saved JSON result
(produced by cooperative_ga_pso.py) and generates:

  1.  Classification report      (accuracy, precision, recall, F1, MCC)
  2.  Confusion matrix heatmap   (PNG)
  3.  ROC curve with AUC         (PNG)
  4.  Precision-Recall curve     (PNG)
  5.  Per-round convergence plot (GA fitness vs PSO fitness over rounds) (PNG)
  6.  Feature selection frequency bar chart (PNG)
  7.  Combined metrics JSON       (all scalar metrics in one file)

All artefacts are saved to:
    results/cooperative_ga_pso/<dataset>_<classifier>_eval/

Usage (standalone)
------------------
    python -m src.evaluate_hybrid \\
        --result results/cooperative_ga_pso/cleveland_SVM_cooperative_ga_pso.json

    # Evaluate all JSON results at once
    python -m src.evaluate_hybrid --result all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — safe on Colab and Windows
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config as cfg
from src.cooperative_ga_pso import (
    build_pipeline,
    load_dataset,
)

# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       130,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
})
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


# ===========================================================================
# Helper — build a Pipeline from a saved JSON result
# ===========================================================================

def _pipeline_from_result(result: dict[str, Any]):
    """Reconstruct the best Pipeline (scaler + classifier) from a JSON result."""
    clf_name = result["classifier_name"]
    hp       = result["best_hyperparams"]
    return build_pipeline(clf_name, hp)


def _subset_X(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Return X restricted to the best features that exist in this dataset."""
    valid = [f for f in feature_names if f in X.columns]
    if not valid:
        raise ValueError(
            f"None of the best features {feature_names} exist in the dataset columns "
            f"{X.columns.tolist()}"
        )
    return X[valid]


# ===========================================================================
# 1. Scalar metrics
# ===========================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Return a dict of all scalar classification metrics."""
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "precision_macro":   float(precision_score(y_true, y_pred, average="macro",  zero_division=0)),
        "recall_macro":      float(recall_score(   y_true, y_pred, average="macro",  zero_division=0)),
        "f1_macro":          float(f1_score(       y_true, y_pred, average="macro",  zero_division=0)),
        "precision_weighted":float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted":   float(recall_score(   y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted":       float(f1_score(       y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc":           float(roc_auc_score(y_true, y_prob)),
        "avg_precision":     float(average_precision_score(y_true, y_prob)),
        "mcc":               float(matthews_corrcoef(y_true, y_pred)),
    }


# ===========================================================================
# 2. Confusion matrix
# ===========================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        linewidths=0.5, linecolor="white", ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    # Dual annotation: count + percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5, i + 0.5,
                f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                ha="center", va="center",
                fontsize=10,
                color="white" if cm[i, j] > cm.max() * 0.6 else "black",
            )
    ax.set_title(title, pad=10, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticklabels(["No Disease (0)", "Disease (1)"], rotation=0)
    ax.set_yticklabels(["No Disease (0)", "Disease (1)"], rotation=90, va="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ===========================================================================
# 3. ROC curve
# ===========================================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc          = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=_PALETTE[0], lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.10, color=_PALETTE[0])
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ===========================================================================
# 4. Precision-Recall curve
# ===========================================================================

def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(recall, precision, color=_PALETTE[1], lw=2, where="post",
            label=f"PR curve (AP = {ap:.4f})")
    ax.fill_between(recall, precision, alpha=0.10, color=_PALETTE[1], step="post")
    ax.axhline(baseline, color="grey", linestyle="--", lw=1,
               label=f"Baseline (prevalence = {baseline:.2f})")
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ===========================================================================
# 5. Convergence plot (per-round GA vs PSO fitness)
# ===========================================================================

def plot_convergence(
    round_logs: list[dict],
    title: str,
    out_path: Path,
) -> None:
    rounds     = [r["round_idx"]         for r in round_logs]
    ga_fitness = [r["ga_best_fitness"]   for r in round_logs]
    pso_fitness= [r["pso_best_fitness"]  for r in round_logs]
    n_feats    = [r["n_selected_features"] for r in round_logs]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(rounds, ga_fitness,  "o-",  color=_PALETTE[0], lw=2, ms=6, label="GA best fitness")
    ax1.plot(rounds, pso_fitness, "s--", color=_PALETTE[1], lw=2, ms=6, label="PSO best fitness")
    ax1.set_xlabel("Cooperative Round")
    ax1.set_ylabel("Best F1 (penalised)")
    ax1.set_title(title, fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right")
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax2 = ax1.twinx()
    ax2.bar(rounds, n_feats, alpha=0.15, color=_PALETTE[2], label="# Features selected")
    ax2.set_ylabel("# Selected features", color=_PALETTE[2])
    ax2.tick_params(axis="y", labelcolor=_PALETTE[2])
    ax2.set_ylim(0, cfg.N_FEATURES * 1.5)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ===========================================================================
# 6. Feature selection frequency bar chart
# ===========================================================================

def plot_feature_frequency(
    round_logs: list[dict],
    all_features: list[str],
    title: str,
    out_path: Path,
) -> None:
    """Count how many rounds each feature was included in the best GA mask."""
    counts: dict[str, int] = {f: 0 for f in all_features}
    for log in round_logs:
        for feat in log.get("selected_features", []):
            if feat in counts:
                counts[feat] += 1

    features = list(counts.keys())
    freqs    = [counts[f] / len(round_logs) * 100 for f in features]   # as %

    # Sort descending for readability
    order = sorted(range(len(freqs)), key=lambda i: freqs[i], reverse=True)
    features = [features[i] for i in order]
    freqs    = [freqs[i]    for i in order]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(features, freqs, color=_PALETTE[0], edgecolor="white", height=0.6)
    ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=9)
    ax.set_xlim(0, 115)
    ax.axvline(50, color="grey", linestyle="--", lw=1, label="50 % threshold")
    ax.set_xlabel("Selection frequency (%)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ===========================================================================
# Master evaluation runner
# ===========================================================================

def evaluate_result(json_path: Path) -> dict[str, Any]:
    """Run full evaluation for one saved cooperative GA-PSO result JSON.

    Parameters
    ----------
    json_path : path to the ``*_cooperative_ga_pso.json`` result file.

    Returns
    -------
    Dict of all scalar metrics.
    """
    with json_path.open(encoding="utf-8") as f:
        result = json.load(f)

    dataset_name  = result["dataset_name"]
    clf_name      = result["classifier_name"]
    best_features = result["best_features"]
    round_logs    = result["round_logs"]

    print(f"\n{'=' * 72}")
    print(f"Evaluating  |  dataset={dataset_name}  |  classifier={clf_name}")
    print(f"Best features ({len(best_features)}): {best_features}")
    print(f"Best hyperparams: {result['best_hyperparams']}")
    print("=" * 72)

    # Output directory
    out_dir = (
        cfg.RESULTS_DIR
        / f"{dataset_name}_{clf_name}_eval"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load data -------------------------------------------------------
    X_full, y = load_dataset(dataset_name)
    X_sub     = _subset_X(X_full, best_features)

    # ----- Cross-validated predictions (no leakage) -------------------------
    pipeline = _pipeline_from_result(result)
    cv        = StratifiedKFold(
        n_splits=cfg.CV_N_SPLITS,
        shuffle=cfg.CV_SHUFFLE,
        random_state=cfg.RANDOM_STATE,
    )

    y_pred = cross_val_predict(pipeline, X_sub, y, cv=cv, method="predict",  n_jobs=-1)
    y_prob = cross_val_predict(pipeline, X_sub, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

    y_np   = y.to_numpy()
    prefix = f"{dataset_name}_{clf_name}"
    title  = f"{clf_name}  |  {dataset_name.capitalize()}"

    # ----- 1. Scalar metrics ------------------------------------------------
    metrics = compute_metrics(y_np, y_pred, y_prob)
    metrics["best_ga_pso_fitness"] = result["best_fitness"]
    metrics["n_best_features"]     = len(best_features)
    metrics["best_features"]       = best_features
    metrics["best_hyperparams"]    = result["best_hyperparams"]

    clf_report_str = classification_report(
        y_np, y_pred,
        target_names=["No Disease", "Disease"],
        zero_division=0,
    )
    print("\nClassification Report (10-fold CV, out-of-fold predictions):")
    print(clf_report_str)

    # Save scalar metrics to JSON
    metrics_path = out_dir / f"{prefix}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
        f.write("\n")

    # Save classification report as CSV
    clf_report_dict = classification_report(
        y_np, y_pred,
        target_names=["No Disease", "Disease"],
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(clf_report_dict).transpose().to_csv(
        out_dir / f"{prefix}_classification_report.csv"
    )

    # ----- 2. Confusion matrix ----------------------------------------------
    plot_confusion_matrix(
        y_np, y_pred,
        title=f"Confusion Matrix — {title}",
        out_path=out_dir / f"{prefix}_confusion_matrix.png",
    )

    # ----- 3. ROC curve -----------------------------------------------------
    plot_roc_curve(
        y_np, y_prob,
        title=f"ROC Curve — {title}",
        out_path=out_dir / f"{prefix}_roc_curve.png",
    )

    # ----- 4. PR curve -------------------------------------------------------
    plot_pr_curve(
        y_np, y_prob,
        title=f"Precision-Recall Curve — {title}",
        out_path=out_dir / f"{prefix}_pr_curve.png",
    )

    # ----- 5. Convergence plot -----------------------------------------------
    plot_convergence(
        round_logs,
        title=f"Cooperative GA-PSO Convergence — {title}",
        out_path=out_dir / f"{prefix}_convergence.png",
    )

    # ----- 6. Feature frequency chart ----------------------------------------
    all_feat_names = list(X_full.columns)
    plot_feature_frequency(
        round_logs,
        all_features=all_feat_names,
        title=f"Feature Selection Frequency — {title}",
        out_path=out_dir / f"{prefix}_feature_frequency.png",
    )

    print(f"\nAll evaluation artefacts saved to: {out_dir}/")
    _print_metrics_table(metrics, prefix)

    return metrics


def _print_metrics_table(metrics: dict[str, Any], prefix: str) -> None:
    """Print a compact metrics table to stdout."""
    rows = [
        ("Accuracy",            f"{metrics['accuracy']:.4f}"),
        ("Precision (weighted)",f"{metrics['precision_weighted']:.4f}"),
        ("Recall (weighted)",   f"{metrics['recall_weighted']:.4f}"),
        ("F1 (weighted)",       f"{metrics['f1_weighted']:.4f}"),
        ("F1 (macro)",          f"{metrics['f1_macro']:.4f}"),
        ("ROC-AUC",             f"{metrics['roc_auc']:.4f}"),
        ("Avg Precision (AP)",  f"{metrics['avg_precision']:.4f}"),
        ("MCC",                 f"{metrics['mcc']:.4f}"),
        ("# Best features",     str(metrics['n_best_features'])),
    ]
    col_w = max(len(r[0]) for r in rows) + 2
    print(f"\n{'Metric':<{col_w}} Value")
    print("-" * (col_w + 8))
    for name, val in rows:
        print(f"  {name:<{col_w - 2}} {val}")
    print()


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate full evaluation reports (confusion matrix, ROC, PR, "
            "convergence, feature frequency) from cooperative GA-PSO result JSONs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result",
        nargs="+",
        default=["all"],
        help=(
            "Path(s) to *_cooperative_ga_pso.json result file(s), "
            "or 'all' to evaluate every JSON in results/cooperative_ga_pso/."
        ),
    )
    args = parser.parse_args()

    if args.result == ["all"] or "all" in args.result:
        json_paths = sorted(cfg.RESULTS_DIR.glob("*_cooperative_ga_pso.json"))
        if not json_paths:
            print(
                f"No result JSONs found in {cfg.RESULTS_DIR}. "
                "Run cooperative_ga_pso.py first."
            )
            return
    else:
        json_paths = [Path(p) for p in args.result]

    all_metrics: list[dict] = []
    for path in json_paths:
        try:
            m = evaluate_result(path)
            all_metrics.append(m)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"Skipping {path.name}: {exc}")

    # Summary table across all evaluated runs
    if len(all_metrics) > 1:
        print("\n" + "=" * 72)
        print("FULL ABLATION EVALUATION SUMMARY")
        print("=" * 72)
        header = (
            f"{'Features':<22} {'#Feats':>7} "
            f"{'Acc':>7} {'F1-W':>7} {'AUC':>7} {'AP':>7} {'MCC':>7}"
        )
        print(header)
        print("-" * len(header))
        for m in all_metrics:
            feat_label = f"{m.get('best_features', ['?'])[0]}…"[:20]
            print(
                f"  {feat_label:<20} {m['n_best_features']:>7} "
                f"{m['accuracy']:>7.4f} {m['f1_weighted']:>7.4f} "
                f"{m['roc_auc']:>7.4f} {m['avg_precision']:>7.4f} "
                f"{m['mcc']:>7.4f}"
            )
        print("=" * 72)


if __name__ == "__main__":
    main()
