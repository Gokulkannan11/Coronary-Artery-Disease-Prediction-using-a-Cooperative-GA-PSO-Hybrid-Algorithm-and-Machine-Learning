"""Train and evaluate the untuned baseline models for available datasets."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.preprocessing import load_and_preprocess
from src.utils import save_results


DATA_RAW_DIR = Path("data") / "raw"
BASELINE_RESULTS_DIR = Path("results") / "baseline"


def results_table(results: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Convert model metrics to a table, excluding nested confusion matrices."""
    table = pd.DataFrame.from_dict(results, orient="index")
    return table.drop(columns=["confusion_matrix"]).rename_axis("model").reset_index()


def save_evaluation_plots(
    dataset_name: str,
    results: dict[str, dict[str, object]],
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
) -> None:
    """Save ROC, precision-recall, and confusion-matrix figures for a dataset."""
    BASELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_names = list(results)

    plt.figure(figsize=(8, 6))
    for model_name in model_names:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test, probabilities[model_name]
        )
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{model_name} (AUC={results[model_name]['auc_roc']:.3f})",
        )
    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curves: {dataset_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(BASELINE_RESULTS_DIR / f"{dataset_name}_roc_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    for model_name in model_names:
        precision, recall, _ = precision_recall_curve(
            y_test, probabilities[model_name]
        )
        plt.plot(recall, precision, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-recall curves: {dataset_name}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(
        BASELINE_RESULTS_DIR / f"{dataset_name}_precision_recall_curve.png",
        dpi=150,
    )
    plt.close()

    figure, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4))
    axes = np.atleast_1d(axes)
    for axis, model_name in zip(axes, model_names):
        sns.heatmap(
            results[model_name]["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axis,
        )
        axis.set_title(model_name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
    figure.suptitle(f"Confusion matrices: {dataset_name}")
    figure.tight_layout()
    figure.savefig(
        BASELINE_RESULTS_DIR / f"{dataset_name}_confusion_matrices.png",
        dpi=150,
    )
    plt.close(figure)


def build_models() -> dict[str, object]:
    """Return the deliberately untuned baseline model definitions."""
    return {
        "SVM (baseline)": SVC(
            kernel="rbf", probability=True, random_state=42
        ),
        "Random Forest (baseline)": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "ANN / MLP (baseline)": MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42
        ),
    }


def evaluate_model(model, x_train, x_test, y_train, y_test) -> dict[str, object]:
    """Fit one model and return test metrics plus five-fold CV accuracy."""
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        clone(model), x_train, y_train, cv=cv, scoring="accuracy"
    )

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "cv_mean_accuracy": float(np.mean(cv_scores)),
        "cv_std_accuracy": float(np.std(cv_scores)),
    }


def run_dataset(csv_path: Path, target_column: str = "target") -> dict[str, dict[str, object]] | None:
    """Run all baselines for one CSV, returning None when it is empty."""
    try:
        prepared = load_and_preprocess(csv_path, target_column=target_column)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        print(f"Skipping {csv_path.name}: file is empty or has no readable CSV data.")
        return None

    x_train, x_test, y_train, y_test, feature_names, _ = prepared
    print(f"\n{'=' * 72}\nDataset: {csv_path.stem}")
    print(f"Records: {len(y_train) + len(y_test)} | Features: {len(feature_names)}")

    results = {}
    predictions = {}
    probabilities = {}
    for model_name, model in build_models().items():
        metrics = evaluate_model(model, x_train, x_test, y_train, y_test)
        results[model_name] = metrics
        model.fit(x_train, y_train)
        predictions[model_name] = model.predict(x_test)
        probabilities[model_name] = model.predict_proba(x_test)[:, 1]

    table = results_table(results)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    save_evaluation_plots(csv_path.stem, results, y_test, predictions, probabilities)

    output_path = BASELINE_RESULTS_DIR / f"{csv_path.stem}_baseline_results.json"
    save_results(results, output_path)
    table.to_csv(output_path.with_suffix(".csv"), index=False)
    print(f"Saved results to {output_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run untuned heart disease baselines.")
    parser.add_argument("--target-column", default="target", help="Name of the target column.")
    args = parser.parse_args()

    if not DATA_RAW_DIR.exists():
        print(f"No raw data directory found at {DATA_RAW_DIR}; nothing to run.")
        return

    csv_paths = sorted(DATA_RAW_DIR.glob("*.csv"))
    if not csv_paths:
        print(f"No CSV datasets found in {DATA_RAW_DIR}; nothing to run.")
        return

    for csv_path in csv_paths:
        try:
            run_dataset(csv_path, target_column=args.target_column)
        except (FileNotFoundError, ValueError, KeyError) as error:
            print(f"Skipping {csv_path.name}: {error}")


if __name__ == "__main__":
    main()
