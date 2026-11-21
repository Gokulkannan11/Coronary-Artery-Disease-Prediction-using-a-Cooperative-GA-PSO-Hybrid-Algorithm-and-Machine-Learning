"""Data loading and preprocessing helpers for baseline experiments."""

from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(
    csv_path: Union[str, Path], target_column: str = "target"
):
    """Load a dataset, impute numeric missing values, scale, and split it.

    The scaler is fit only on the training features to avoid leaking test-set
    information into the training pipeline.
    """
    csv_path = Path(csv_path)
    data = pd.read_csv(csv_path, comment="#")

    if target_column not in data.columns:
        raise ValueError(
            f"Target column {target_column!r} was not found in {csv_path}. "
            f"Available columns: {list(data.columns)}"
        )

    missing_values = int(data.isna().sum().sum())
    if missing_values:
        print(f"{csv_path.name}: found {missing_values} missing value(s); imputing numeric columns with medians.")
        numeric_columns = data.select_dtypes(include="number").columns
        data.loc[:, numeric_columns] = data.loc[:, numeric_columns].fillna(
            data.loc[:, numeric_columns].median()
        )
    else:
        print(f"{csv_path.name}: no missing values found.")

    features = data.drop(columns=[target_column])
    target = data[target_column]
    non_numeric_columns = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_columns:
        raise ValueError(
            "All feature columns must be numeric for StandardScaler. "
            f"Non-numeric columns: {non_numeric_columns}"
        )

    if target.isna().any():
        raise ValueError("The target column contains missing values and cannot be imputed safely.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return (
        x_train_scaled,
        x_test_scaled,
        y_train,
        y_test,
        features.columns.tolist(),
        scaler,
    )
