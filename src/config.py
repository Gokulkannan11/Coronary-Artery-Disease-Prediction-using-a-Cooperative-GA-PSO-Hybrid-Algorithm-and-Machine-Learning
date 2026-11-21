"""Centralized configuration for the Cooperative GA-PSO hybrid algorithm.

All hyperparameters, random states, cross-validation settings, dataset paths,
and algorithm-specific knobs live here.  Import this module anywhere in the
project rather than hard-coding values in individual scripts.

Google Drive path (Colab):
    /content/drive/MyDrive/Heart_Disease_Hybrid_EA/src/
Local path (fallback):
    data/raw/
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Dataset paths
# Google Colab: mount Drive first, then these paths resolve automatically.
# Local:        paths fall back to the project's data/raw/ directory.
# ---------------------------------------------------------------------------
_COLAB_ROOT = Path("/content/drive/MyDrive/Heart_Disease_Hybrid_EA/src")
_LOCAL_ROOT = Path("data") / "raw"

# Prefer Colab paths when running on Drive; otherwise fall back to local.
DATA_ROOT: Path = _COLAB_ROOT if _COLAB_ROOT.exists() else _LOCAL_ROOT

DATASET_PATHS: dict[str, Path] = {
    "cleveland": DATA_ROOT / "Heart_disease_cleveland_new.csv",
    "statlog":   DATA_ROOT / "statlog_heart.csv",
    "hungarian": DATA_ROOT / "hungarian_heart_disease.csv",
}

# Name of the binary target column (0 = no disease, 1 = disease).
TARGET_COLUMN: str = "target"

# ---------------------------------------------------------------------------
# Results output directory
# ---------------------------------------------------------------------------
RESULTS_DIR: Path = Path("results") / "cooperative_ga_pso"

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
CV_N_SPLITS:   int = 10        # 10-fold stratified CV
CV_SHUFFLE:    bool = True
CV_SCORING:    str  = "f1"     # optimisation objective inside fitness function

# ---------------------------------------------------------------------------
# Feature space
# ---------------------------------------------------------------------------
# The 13 standard UCI Cleveland clinical features (column names must match CSV).
FEATURE_NAMES: list[str] = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]
N_FEATURES: int = len(FEATURE_NAMES)   # 13

# Penalty applied to the fitness score for each selected feature
# (encourages sparser, more interpretable subsets).
FEATURE_PENALTY: float = 0.005

# Minimum number of features that must remain active in any individual.
MIN_FEATURES: int = 2

# ---------------------------------------------------------------------------
# Cooperative loop schedule
# ---------------------------------------------------------------------------
# Total number of cooperative rounds (GA generation batch <-> PSO iteration).
N_ROUNDS: int = 10

# Number of GA generations executed per cooperative round.
GA_GENS_PER_ROUND: int = 5

# Number of PSO iterations executed per cooperative round.
PSO_ITERS_PER_ROUND: int = 20

# ---------------------------------------------------------------------------
# Genetic Algorithm (DEAP)
# ---------------------------------------------------------------------------
GA_POP_SIZE:       int   = 50     # number of individuals
GA_CX_PROB:        float = 0.7    # crossover probability
GA_MUT_PROB:       float = 0.2    # mutation probability per individual
GA_INDPB:          float = 0.1    # per-bit flip probability inside mutFlipBit
GA_TOURNAMENT_SIZE: int  = 3      # tournament selection size

# ---------------------------------------------------------------------------
# Particle Swarm Optimisation (PySwarms)
# ---------------------------------------------------------------------------
PSO_N_PARTICLES: int   = 30       # swarm size
PSO_COGNITIVE:   float = 2.0      # c1 - personal best weight (higher → more exploration)
PSO_SOCIAL:      float = 1.5      # c2 - global best weight
PSO_INERTIA:     float = 0.4      # w  - lower inertia → stronger local search, less coast

# ---------------------------------------------------------------------------
# Classifier hyperparameter search bounds
# Each row:  [lower_bound, upper_bound]
# Dimensions are concatenated in this order for the PSO position vector:
#   SVM  : [log10(C), log10(gamma)]          dims 0-1
#   RF   : [n_estimators, max_depth]          dims 2-3
#   XGB  : [n_estimators, max_depth,          dims 4-6
#            learning_rate]
# ---------------------------------------------------------------------------
PSO_BOUNDS_LOW: list[float] = [
    -3.0,   # SVM  log10(C)        -> C in [0.001, 1000]
    -4.0,   # SVM  log10(gamma)    -> gamma in [0.0001, 1]
    10.0,   # RF   n_estimators    -> [10, 500]
     2.0,   # RF   max_depth       -> [2, 30]
    10.0,   # XGB  n_estimators    -> [10, 500]
     2.0,   # XGB  max_depth       -> [2, 10]
     0.01,  # XGB  learning_rate   -> [0.01, 0.5]
]

PSO_BOUNDS_HIGH: list[float] = [
     3.0,   # SVM  log10(C)
     0.0,   # SVM  log10(gamma)
   500.0,   # RF   n_estimators
    30.0,   # RF   max_depth
   500.0,   # XGB  n_estimators
    10.0,   # XGB  max_depth
     0.5,   # XGB  learning_rate
]

PSO_DIMENSIONS: int = len(PSO_BOUNDS_LOW)  # 7

# ---------------------------------------------------------------------------
# Classifier selection
# Names must match keys used in cooperative_ga_pso.py build_classifiers().
# ---------------------------------------------------------------------------
ACTIVE_CLASSIFIERS: list[str] = ["SVM", "RandomForest", "XGBoost"]

# ---------------------------------------------------------------------------
# Logging & verbosity
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
VERBOSE:   bool = True   # print round-by-round progress
