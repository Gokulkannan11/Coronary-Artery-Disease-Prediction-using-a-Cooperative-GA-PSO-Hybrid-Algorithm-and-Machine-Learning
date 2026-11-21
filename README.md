# Hybrid Evolutionary Optimization + ML for Heart Disease Prediction

This MTech CS project studies heart disease prediction using machine learning and a planned hybrid evolutionary optimization pipeline. The research will compare untuned baseline models with GA-optimized, PSO-optimized, and cooperative GA-PSO approaches across multiple heart disease datasets. This repository currently implements phase 1 only: baseline models and the supporting data infrastructure.

## Setup

The project targets Python 3.10 or newer.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source venv/bin/activate`.

Place the clean Cleveland CSV at `data/raw/Heart_disease_cleveland_new.csv`. It must contain a binary `target` column and numeric feature columns. The Statlog and Z-Alizadeh Sani files are currently empty placeholders and are skipped automatically.

## Run Baselines

From the repository root, activate the virtual environment and run:

```powershell
python -m src.baseline_models
```

The runner discovers every CSV in `data/raw/`, skips empty or unreadable placeholders with a clear message, prints test and cross-validation metrics as a table, and writes JSON, CSV, ROC curve, precision-recall curve, and confusion-matrix results to `results/baseline/`.

For a dataset with a different target column name:

```powershell
python -m src.baseline_models --target-column disease
```

Run the tests with:

```powershell
pytest
```

## Project Roadmap

The following research stages are planned and are **NOT YET IMPLEMENTED**:

- GA-only feature/model optimization
- PSO-only feature/model optimization
- Sequential GA-PSO hybrid optimization
- Cooperative GA-PSO optimization
- Baseline versus optimized ablation study
- SHAP explainability

No evolutionary optimization, XGBoost, or SHAP dependency is included in this phase.
