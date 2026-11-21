from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing import load_and_preprocess


def test_load_and_preprocess_returns_expected_shapes(tmp_path: Path):
    data = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_b": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    csv_path = tmp_path / "synthetic.csv"
    data.to_csv(csv_path, index=False)

    x_train, x_test, y_train, y_test, feature_names, scaler = load_and_preprocess(csv_path)

    assert x_train.shape == (8, 2)
    assert x_test.shape == (2, 2)
    assert y_train.shape == (8,)
    assert y_test.shape == (2,)
    assert feature_names == ["feature_a", "feature_b"]
    assert scaler.mean_.shape == (2,)


def test_baseline_runner_handles_cleveland_when_present():
    csv_path = Path("data/raw/Heart_disease_cleveland_new.csv")
    if not csv_path.exists():
        pytest.skip("Cleveland dataset is not present locally.")

    from src.baseline_models import run_dataset

    results = run_dataset(csv_path)
    assert results is not None
    assert set(results) == {
        "SVM (baseline)",
        "Random Forest (baseline)",
        "ANN / MLP (baseline)",
    }
