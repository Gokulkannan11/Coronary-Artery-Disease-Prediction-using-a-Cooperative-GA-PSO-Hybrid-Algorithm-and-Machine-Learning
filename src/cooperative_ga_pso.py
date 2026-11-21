"""Cooperative GA-PSO Hybrid Algorithm for Coronary Artery Disease Prediction.

Architecture
------------
This script implements a **scheduled, two-way cooperative feedback loop**
between two optimisers:

  GA  (DEAP)      - Discrete feature-subset selection over the 13 UCI
                    clinical features.  Each individual is a binary
                    chromosome [0/1]^13.

  PSO (PySwarms)  - Continuous hyperparameter optimisation for SVM,
                    Random Forest, and XGBoost across a 7-dimensional
                    real-valued search space.

Cooperative Exchange Schedule
------------------------------
Every cooperative round:
  1. GA evolves for GA_GENS_PER_ROUND generations using the *current best
     hyperparameters* provided by the PSO's global-best particle.
  2. The GA's best feature mask is pushed back to the PSO so that all
     PSO particles evaluate their classifier on the *current best features*
     found by the GA.
  3. The PSO executes PSO_ITERS_PER_ROUND iterations using the new feature
     mask; its new global-best hyperparameters are fed forward to the GA.
  4. Repeat for N_ROUNDS rounds.

This bidirectional coupling prevents either optimiser from converging in
isolation and forms the core novelty of the architecture.

Anti-leakage
------------
StandardScaler + classifier are always wrapped inside a scikit-learn
Pipeline so that scaling is fit ONLY on each CV training fold.

Data-leakage prevention is therefore guaranteed at the cross-validation level.

Usage
-----
    # Local run (uses data/raw/*.csv)
    python -m src.cooperative_ga_pso

    # Google Colab
    # 1. Mount Drive: from google.colab import drive; drive.mount('/content/drive')
    # 2. python -m src.cooperative_ga_pso --dataset cleveland

Dependencies
------------
    pip install deap pyswarms xgboost scikit-learn pandas numpy
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGBOOST_AVAILABLE = False
    warnings.warn(
        "XGBoost is not installed.  The 'XGBoost' classifier will be skipped. "
        "Install it with:  pip install xgboost",
        stacklevel=2,
    )

# DEAP imports
from deap import algorithms, base, creator, tools

# PySwarms import
try:
    import pyswarms as ps
    from pyswarms.single import GlobalBestPSO
    _PYSWARMS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYSWARMS_AVAILABLE = False
    raise ImportError(
        "PySwarms is required.  Install it with:  pip install pyswarms"
    )

# Local config (all magic numbers live in config.py).
from src import config as cfg

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=getattr(logging, cfg.LOG_LEVEL, logging.INFO),
)
logger = logging.getLogger(__name__)

# Suppress verbose scikit-learn / PySwarms warnings during CV sweeps.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", module="pyswarms")

# ---------------------------------------------------------------------------
# DEAP creator setup (module-level, guarded to avoid double-registration)
# ---------------------------------------------------------------------------
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


# ===========================================================================
# Data Structures
# ===========================================================================

@dataclass
class RoundLog:
    """Snapshot of metrics captured at the end of one cooperative round."""
    round_idx:         int
    classifier_name:   str
    dataset_name:      str
    ga_best_fitness:   float
    pso_best_fitness:  float
    n_selected_features: int
    selected_features: list[str]
    best_hyperparams:  dict[str, Any]
    wall_time_s:       float


@dataclass
class RunResult:
    """Aggregated output from a full cooperative run on one dataset/classifier pair."""
    dataset_name:     str
    classifier_name:  str
    best_fitness:     float
    best_features:    list[str]
    best_hyperparams: dict[str, Any]
    round_logs:       list[RoundLog] = field(default_factory=list)
    total_wall_time_s: float = 0.0


# ===========================================================================
# Data Loading
# ===========================================================================

def load_dataset(dataset_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a named dataset, imputing numeric NaNs with column medians.

    Parameters
    ----------
    dataset_name:
        One of the keys in ``cfg.DATASET_PATHS`` ('cleveland', 'statlog',
        'hungarian').

    Returns
    -------
    X : pd.DataFrame  — features (all numeric, NaN-free)
    y : pd.Series     — binary target (0 / 1)
    """
    path = cfg.DATASET_PATHS.get(dataset_name)
    if path is None:
        raise KeyError(
            f"Unknown dataset {dataset_name!r}. "
            f"Available: {list(cfg.DATASET_PATHS)}"
        )
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "If running on Colab, mount your Drive first:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')"
        )

    df = pd.read_csv(path, comment="#")

    if cfg.TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{cfg.TARGET_COLUMN}' not found in {path}. "
            f"Columns present: {df.columns.tolist()}"
        )

    # Impute numeric NaNs with column medians.
    numeric_cols = df.select_dtypes(include="number").columns
    n_missing = int(df[numeric_cols].isna().sum().sum())
    if n_missing:
        logger.info("  Imputing %d missing value(s) with column medians.", n_missing)
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Resolve actual feature columns (only use FEATURE_NAMES that exist in the CSV).
    available_features = [f for f in cfg.FEATURE_NAMES if f in df.columns]
    if len(available_features) < cfg.MIN_FEATURES:
        raise ValueError(
            f"Only {len(available_features)} of the expected {cfg.N_FEATURES} "
            f"features found in {path.name}. Cannot continue."
        )

    X = df[available_features].copy()
    y = df[cfg.TARGET_COLUMN].copy()

    # Binarise target: any non-zero value → 1 (handles multi-class UCI format).
    y = (y > 0).astype(int)

    logger.info(
        "  Loaded '%s': %d samples, %d features, %.1f%% positive.",
        dataset_name,
        len(y),
        X.shape[1],
        100 * y.mean(),
    )
    return X, y


# ===========================================================================
# Classifier Builders
# ===========================================================================

def _decode_pso_position(position: np.ndarray) -> dict[str, dict[str, Any]]:
    """Decode a 7-D PSO position vector into per-classifier hyperparameter dicts.

    Layout:
        [0]  SVM  log10(C)
        [1]  SVM  log10(gamma)
        [2]  RF   n_estimators   (int)
        [3]  RF   max_depth      (int)
        [4]  XGB  n_estimators   (int)
        [5]  XGB  max_depth      (int)
        [6]  XGB  learning_rate
    """
    return {
        "SVM": {
            "C":     float(10 ** np.clip(position[0], cfg.PSO_BOUNDS_LOW[0], cfg.PSO_BOUNDS_HIGH[0])),
            "gamma": float(10 ** np.clip(position[1], cfg.PSO_BOUNDS_LOW[1], cfg.PSO_BOUNDS_HIGH[1])),
        },
        "RandomForest": {
            "n_estimators": int(np.clip(round(position[2]), cfg.PSO_BOUNDS_LOW[2], cfg.PSO_BOUNDS_HIGH[2])),
            "max_depth":    int(np.clip(round(position[3]), cfg.PSO_BOUNDS_LOW[3], cfg.PSO_BOUNDS_HIGH[3])),
        },
        "XGBoost": {
            "n_estimators":  int(np.clip(round(position[4]), cfg.PSO_BOUNDS_LOW[4], cfg.PSO_BOUNDS_HIGH[4])),
            "max_depth":     int(np.clip(round(position[5]), cfg.PSO_BOUNDS_LOW[5], cfg.PSO_BOUNDS_HIGH[5])),
            "learning_rate": float(np.clip(position[6], cfg.PSO_BOUNDS_LOW[6], cfg.PSO_BOUNDS_HIGH[6])),
        },
    }


def build_pipeline(classifier_name: str, hyperparams: dict[str, Any]) -> Pipeline:
    """Return a ``StandardScaler`` + classifier Pipeline for the given name and params.

    The scaler is intentionally inside the pipeline so that it is re-fit on
    every CV training fold, preventing data leakage.
    """
    if classifier_name == "SVM":
        clf = SVC(
            kernel="rbf",
            C=hyperparams.get("C", 1.0),
            gamma=hyperparams.get("gamma", "scale"),
            probability=True,    # required for predict_proba in evaluation
            random_state=cfg.RANDOM_STATE,
        )
    elif classifier_name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", None),
            random_state=cfg.RANDOM_STATE,
            n_jobs=-1,
        )
    elif classifier_name == "XGBoost":
        if not _XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost is not installed.")
        clf = XGBClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", 6),
            learning_rate=hyperparams.get("learning_rate", 0.1),
            random_state=cfg.RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_name!r}")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


# ===========================================================================
# Shared Fitness Evaluation
# ===========================================================================

def evaluate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    feature_mask: list[int],
) -> float:
    """Return the mean 10-fold stratified CV F1 score on the selected features.

    A sparse-feature penalty is subtracted so the GA is nudged toward compact
    subsets.  The result is in [0, 1] (higher is better).

    Parameters
    ----------
    pipeline     : fitted Pipeline (scaler inside — no leakage)
    X            : full feature DataFrame
    y            : target Series
    feature_mask : binary list of length N_FEATURES

    Returns
    -------
    float  — penalised F1 score
    """
    selected_indices = [i for i, bit in enumerate(feature_mask) if bit == 1]
    if len(selected_indices) < cfg.MIN_FEATURES:
        return 0.0   # infeasible individual → zero fitness

    X_sub = X.iloc[:, selected_indices]

    cv = StratifiedKFold(
        n_splits=cfg.CV_N_SPLITS,
        shuffle=cfg.CV_SHUFFLE,
        random_state=cfg.RANDOM_STATE,
    )
    scores = cross_val_score(
        pipeline, X_sub, y,
        cv=cv,
        scoring=cfg.CV_SCORING,
        n_jobs=-1,
    )
    raw_score = float(np.mean(scores))
    penalty   = cfg.FEATURE_PENALTY * len(selected_indices)
    return max(0.0, raw_score - penalty)


# ===========================================================================
# GA Component (DEAP)
# ===========================================================================

class GAFeatureSelector:
    """Genetic algorithm for discrete feature-subset selection using DEAP.

    The chromosome is a binary list of length ``n_features``.
    Fitness is the penalised 10-fold stratified CV F1 score evaluated with the
    hyperparameters currently shared by the PSO.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int,
        classifier_name: str,
    ) -> None:
        self.X               = X
        self.y               = y
        self.n_features      = n_features
        self.classifier_name = classifier_name

        # Hyperparameters shared by the PSO; initialised to safe defaults.
        self._shared_hyperparams: dict[str, Any] = {}

        # Current best individual (binary list) and its fitness.
        self.best_mask:    list[int] = [1] * n_features
        self.best_fitness: float     = 0.0

        # DEAP toolbox
        self._toolbox = self._build_toolbox()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def receive_hyperparams(self, hyperparams: dict[str, Any]) -> None:
        """Accept hyperparameters from the PSO and store them for fitness eval."""
        self._shared_hyperparams = hyperparams
        logger.debug("GA received new hyperparams: %s", hyperparams)

    def run(self, population: list | None, n_generations: int) -> list:
        """Evolve the population for ``n_generations`` generations.

        Parameters
        ----------
        population : existing DEAP population or None (will be seeded fresh).
        n_generations : number of generations to run.

        Returns
        -------
        Updated population list.
        """
        if population is None:
            population = self._toolbox.population(n=cfg.GA_POP_SIZE)

        # Evaluate individuals whose fitness is not yet valid.
        invalid = [ind for ind in population if not ind.fitness.valid]
        fitnesses = list(map(self._toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        # Run the evolutionary loop manually so we can inspect statistics.
        for _ in range(n_generations):
            offspring = algorithms.varAnd(
                population, self._toolbox,
                cxpb=cfg.GA_CX_PROB,
                mutpb=cfg.GA_MUT_PROB,
            )
            # Evaluate offspring.
            invalid_off = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(self._toolbox.evaluate, invalid_off))
            for ind, fit in zip(invalid_off, fits):
                ind.fitness.values = fit

            population = self._toolbox.select(offspring, k=cfg.GA_POP_SIZE)

        # Track global best across all generations.
        best_ind = tools.selBest(population, k=1)[0]
        if best_ind.fitness.values[0] > self.best_fitness:
            self.best_fitness = best_ind.fitness.values[0]
            self.best_mask    = list(best_ind)

        return population

    def get_best_mask(self) -> list[int]:
        """Return the best feature mask discovered so far."""
        return list(self.best_mask)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate(self, individual: list[int]) -> tuple[float]:
        """DEAP-compatible fitness function."""
        pipeline = build_pipeline(self.classifier_name, self._shared_hyperparams)
        score = evaluate_pipeline(pipeline, self.X, self.y, individual)
        return (score,)

    def _build_toolbox(self) -> base.Toolbox:
        toolbox = base.Toolbox()

        # Gene: random 0/1
        toolbox.register("attr_bool", random.randint, 0, 1)

        # Individual: binary list of length n_features
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_bool,
            n=self.n_features,
        )

        # Population
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Operators
        toolbox.register("evaluate",  self._evaluate)
        toolbox.register("mate",      tools.cxTwoPoint)
        toolbox.register(
            "mutate",
            tools.mutFlipBit,
            indpb=cfg.GA_INDPB,
        )
        toolbox.register(
            "select",
            tools.selTournament,
            tournsize=cfg.GA_TOURNAMENT_SIZE,
        )
        return toolbox


# ===========================================================================
# PSO Component (PySwarms)
# ===========================================================================

class PSOHyperparamOptimiser:
    """PSO-based continuous hyperparameter optimiser using PySwarms.

    The swarm searches the 7-D space defined in ``config.py``.
    Each particle's fitness is the penalised 10-fold F1 evaluated on the
    *current best feature mask* shared by the GA.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        classifier_name: str,
    ) -> None:
        self.X               = X
        self.y               = y
        self.classifier_name = classifier_name

        # Feature mask shared by the GA; initialised to select all features.
        self._shared_mask: list[int] = [1] * cfg.N_FEATURES

        # Best decoded hyperparams and their cost (PySwarms minimises cost).
        self.best_hyperparams: dict[str, Any] = {}
        self.best_fitness:     float = 0.0
        self.best_cost:        float = np.inf

        # Raw best position vector (7-D); used to warm-start subsequent rounds.
        self._best_pos: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def receive_feature_mask(self, mask: list[int]) -> None:
        """Accept a feature mask from the GA and store it for fitness eval."""
        self._shared_mask = mask
        logger.debug(
            "PSO received new feature mask: %d/%d features selected.",
            sum(mask), len(mask),
        )

    def run(self, n_iterations: int) -> None:
        """Advance the swarm for ``n_iterations`` steps.

        PySwarms 1.3.0: ``GlobalBestPSO.optimize`` returns ``(best_cost, best_pos)``
        directly.  We warm-start subsequent rounds via the ``init_pos`` constructor
        parameter — a full ``(n_particles × dims)`` position matrix seeded from the
        previous round's global best with small random perturbations.
        """
        # Build warm-start position matrix for round continuity.
        init_pos: np.ndarray | None = None
        if self._best_pos is not None:
            low  = np.array(cfg.PSO_BOUNDS_LOW,  dtype=float)
            high = np.array(cfg.PSO_BOUNDS_HIGH, dtype=float)
            # Tile best position across all particles, then add bounded noise.
            # 20 % of the range gives enough spread to escape the convergence basin.
            init_pos = np.tile(self._best_pos, (cfg.PSO_N_PARTICLES, 1))
            noise    = np.random.uniform(-0.20, 0.20, init_pos.shape) * (high - low)
            init_pos = np.clip(init_pos + noise, low, high)
            init_pos[0] = self._best_pos   # keep one exact elite particle

        self._init_optimizer(init_pos=init_pos)

        # optimize() returns (best_cost: float, best_pos: ndarray) in PySwarms 1.x
        best_cost, best_pos = self._optimizer.optimize(
            self._cost_fn,
            iters=n_iterations,
            verbose=False,
        )

        if best_cost < self.best_cost:
            self.best_cost        = best_cost
            self.best_fitness     = 1.0 - best_cost   # convert minimise → maximise
            self._best_pos        = best_pos.copy()
            decoded               = _decode_pso_position(best_pos)
            self.best_hyperparams = decoded[self.classifier_name]

        logger.debug(
            "PSO best cost %.4f  |  decoded: %s",
            self.best_cost,
            self.best_hyperparams,
        )

    def get_best_hyperparams(self) -> dict[str, Any]:
        """Return the best hyperparameters discovered so far (for this classifier)."""
        return dict(self.best_hyperparams)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cost_fn(self, positions: np.ndarray) -> np.ndarray:
        """PySwarms cost function: evaluate each particle in the swarm.

        PySwarms passes the entire swarm matrix at once (n_particles × dims).
        We return a 1-D cost array (lower is better) by computing 1 - F1.

        The shared feature mask (updated by GA) is used for all evaluations.
        """
        costs = np.empty(positions.shape[0])
        for i, pos in enumerate(positions):
            decoded  = _decode_pso_position(pos)
            hp       = decoded[self.classifier_name]
            pipeline = build_pipeline(self.classifier_name, hp)
            score    = evaluate_pipeline(pipeline, self.X, self.y, self._shared_mask)
            costs[i] = 1.0 - score   # minimise cost
        return costs

    def _init_optimizer(self, init_pos: np.ndarray | None = None) -> None:
        """(Re-)initialise the GlobalBestPSO with current bounds and options.

        Parameters
        ----------
        init_pos : optional (n_particles × dims) array passed to ``GlobalBestPSO``
            to warm-start the swarm.  If None, PySwarms samples positions
            uniformly from the bounds (cold start).
        """
        bounds = (
            np.array(cfg.PSO_BOUNDS_LOW,  dtype=float),
            np.array(cfg.PSO_BOUNDS_HIGH, dtype=float),
        )
        options = {
            "c1": cfg.PSO_COGNITIVE,
            "c2": cfg.PSO_SOCIAL,
            "w":  cfg.PSO_INERTIA,
        }
        self._optimizer = GlobalBestPSO(
            n_particles=cfg.PSO_N_PARTICLES,
            dimensions=cfg.PSO_DIMENSIONS,
            options=options,
            bounds=bounds,
            init_pos=init_pos,   # None → random uniform; array → warm start
        )


def _encode_hyperparams_to_position(
    classifier_name: str,
    hyperparams: dict[str, Any],
) -> np.ndarray:
    """Re-encode decoded hyperparams back to a 7-D PSO position vector.

    Used for warm-starting the swarm from the previous round's best position.
    Dimensions not belonging to ``classifier_name`` are set to the midpoint of
    their bounds so they do not bias other particles.
    """
    low  = np.array(cfg.PSO_BOUNDS_LOW,  dtype=float)
    high = np.array(cfg.PSO_BOUNDS_HIGH, dtype=float)
    pos  = (low + high) / 2.0   # default: midpoint

    if classifier_name == "SVM":
        pos[0] = np.log10(max(hyperparams.get("C", 1.0), 1e-10))
        pos[1] = np.log10(max(hyperparams.get("gamma", 0.01), 1e-10))
    elif classifier_name == "RandomForest":
        pos[2] = float(hyperparams.get("n_estimators", 100))
        pos[3] = float(hyperparams.get("max_depth", 10))
    elif classifier_name == "XGBoost":
        pos[4] = float(hyperparams.get("n_estimators", 100))
        pos[5] = float(hyperparams.get("max_depth", 6))
        pos[6] = float(hyperparams.get("learning_rate", 0.1))

    return np.clip(pos, low, high)


# ===========================================================================
# Cooperative Loop
# ===========================================================================

def run_cooperative_loop(
    X: pd.DataFrame,
    y: pd.Series,
    classifier_name: str,
    dataset_name: str,
    n_rounds: int = cfg.N_ROUNDS,
    ga_gens_per_round: int = cfg.GA_GENS_PER_ROUND,
    pso_iters_per_round: int = cfg.PSO_ITERS_PER_ROUND,
) -> RunResult:
    """Execute the full cooperative GA <-> PSO feedback loop.

    Round-by-Round Exchange Protocol
    ---------------------------------
    Round k:
      Step 1 — PSO → GA:
          GA receives the PSO's current global-best hyperparameters.
          GA runs for ``ga_gens_per_round`` generations.

      Step 2 — GA → PSO:
          PSO receives the GA's current best feature mask.
          PSO runs for ``pso_iters_per_round`` iterations.

      Logging:
          Round metrics are recorded and printed if ``cfg.VERBOSE``.

    Parameters
    ----------
    X, y              : full feature matrix and target.
    classifier_name   : 'SVM', 'RandomForest', or 'XGBoost'.
    dataset_name      : used only for logging / result labels.
    n_rounds          : total cooperative rounds.
    ga_gens_per_round : GA generations per round.
    pso_iters_per_round : PSO iterations per round.

    Returns
    -------
    RunResult with the best features, best hyperparameters, and per-round logs.
    """
    logger.info(
        "\n%s\nCooperative GA-PSO | Dataset: %-12s | Classifier: %s\n%s",
        "=" * 72, dataset_name, classifier_name, "=" * 72,
    )

    # Resolve actual feature columns present in X.
    actual_features = list(X.columns)
    n_features      = len(actual_features)

    # Instantiate the two optimisers.
    ga  = GAFeatureSelector(X, y, n_features, classifier_name)
    pso = PSOHyperparamOptimiser(X, y, classifier_name)

    ga_population: list | None = None   # carry population across rounds
    round_logs:    list[RoundLog] = []
    run_start      = time.perf_counter()

    for round_idx in range(1, n_rounds + 1):
        t0 = time.perf_counter()

        logger.info("Round %2d/%d", round_idx, n_rounds)

        # ------------------------------------------------------------------
        # Step 1: PSO → GA  (hyperparameter injection)
        # ------------------------------------------------------------------
        current_hyperparams = pso.get_best_hyperparams()
        ga.receive_hyperparams(current_hyperparams)

        ga_population = ga.run(ga_population, n_generations=ga_gens_per_round)
        ga_mask   = ga.get_best_mask()
        ga_fitness = ga.best_fitness

        logger.info(
            "  GA  best fitness=%.4f | selected %d/%d features",
            ga_fitness,
            sum(ga_mask),
            n_features,
        )

        # ------------------------------------------------------------------
        # Step 2: GA → PSO  (feature mask injection)
        # ------------------------------------------------------------------
        pso.receive_feature_mask(ga_mask)
        pso.run(n_iterations=pso_iters_per_round)

        pso_fitness    = pso.best_fitness
        best_hp        = pso.get_best_hyperparams()

        logger.info(
            "  PSO best fitness=%.4f | hyperparams: %s",
            pso_fitness,
            best_hp,
        )

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        selected_names = [
            actual_features[i]
            for i, bit in enumerate(ga_mask)
            if bit == 1
        ]
        round_log = RoundLog(
            round_idx=round_idx,
            classifier_name=classifier_name,
            dataset_name=dataset_name,
            ga_best_fitness=ga_fitness,
            pso_best_fitness=pso_fitness,
            n_selected_features=sum(ga_mask),
            selected_features=selected_names,
            best_hyperparams=deepcopy(best_hp),
            wall_time_s=time.perf_counter() - t0,
        )
        round_logs.append(round_log)

        if cfg.VERBOSE:
            _print_round_summary(round_log)

    # ------------------------------------------------------------------
    # Assemble final result
    # ------------------------------------------------------------------
    final_mask = ga.get_best_mask()
    final_features = [
        actual_features[i]
        for i, bit in enumerate(final_mask)
        if bit == 1
    ]
    final_hyperparams = pso.get_best_hyperparams()
    final_fitness     = max(ga.best_fitness, pso.best_fitness)

    return RunResult(
        dataset_name=dataset_name,
        classifier_name=classifier_name,
        best_fitness=final_fitness,
        best_features=final_features,
        best_hyperparams=final_hyperparams,
        round_logs=round_logs,
        total_wall_time_s=time.perf_counter() - run_start,
    )


# ===========================================================================
# Reporting
# ===========================================================================

def _print_round_summary(log: RoundLog) -> None:
    """Print a formatted one-line summary of a cooperative round."""
    print(
        f"  [R{log.round_idx:02d}] "
        f"GA={log.ga_best_fitness:.4f}  "
        f"PSO={log.pso_best_fitness:.4f}  "
        f"feats={log.n_selected_features}  "
        f"time={log.wall_time_s:.1f}s  "
        f"hp={log.best_hyperparams}"
    )


def print_final_report(result: RunResult) -> None:
    """Print the final optimisation summary for one dataset/classifier pair."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"FINAL RESULT  |  {result.dataset_name}  |  {result.classifier_name}")
    print(sep)
    print(f"  Best fitness (F1):     {result.best_fitness:.4f}")
    print(f"  Selected features [{len(result.best_features)}]: {result.best_features}")
    print(f"  Best hyperparameters:  {result.best_hyperparams}")
    print(f"  Total wall time:       {result.total_wall_time_s:.1f}s")
    print(sep)


def save_run_result(result: RunResult) -> Path:
    """Persist the run result as a JSON file in the results directory.

    Returns the path of the saved file.
    """
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{result.dataset_name}_{result.classifier_name}_cooperative_ga_pso.json"
    )
    out_path = cfg.RESULTS_DIR / filename

    # Convert dataclasses to plain dicts for JSON serialisation.
    payload = {
        "dataset_name":      result.dataset_name,
        "classifier_name":   result.classifier_name,
        "best_fitness":      result.best_fitness,
        "best_features":     result.best_features,
        "best_hyperparams":  result.best_hyperparams,
        "total_wall_time_s": result.total_wall_time_s,
        "round_logs": [asdict(log) for log in result.round_logs],
        "config": {
            "RANDOM_STATE":       cfg.RANDOM_STATE,
            "CV_N_SPLITS":        cfg.CV_N_SPLITS,
            "CV_SCORING":         cfg.CV_SCORING,
            "N_ROUNDS":           cfg.N_ROUNDS,
            "GA_GENS_PER_ROUND":  cfg.GA_GENS_PER_ROUND,
            "PSO_ITERS_PER_ROUND": cfg.PSO_ITERS_PER_ROUND,
            "GA_POP_SIZE":        cfg.GA_POP_SIZE,
            "PSO_N_PARTICLES":    cfg.PSO_N_PARTICLES,
            "FEATURE_PENALTY":    cfg.FEATURE_PENALTY,
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")

    logger.info("Results saved to %s", out_path)
    return out_path


# ===========================================================================
# Entry Point
# ===========================================================================

def main() -> None:
    """CLI entry point for the cooperative GA-PSO optimiser.

    Examples
    --------
    # Run all classifiers on the Cleveland dataset
    python -m src.cooperative_ga_pso --dataset cleveland

    # Run only SVM on both Cleveland and Statlog
    python -m src.cooperative_ga_pso --dataset cleveland statlog --classifier SVM

    # Full ablation on all datasets and classifiers
    python -m src.cooperative_ga_pso --dataset all --classifier all
    """
    parser = argparse.ArgumentParser(
        description=(
            "Cooperative GA-PSO Hybrid Algorithm for Coronary Artery Disease Prediction. "
            "GA performs feature subset selection; PSO performs hyperparameter tuning. "
            "Both optimisers exchange information on a defined schedule."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["cleveland"],
        choices=[*list(cfg.DATASET_PATHS.keys()), "all"],
        help=(
            "Dataset(s) to run. Use 'all' to run Cleveland, Statlog, and Hungarian."
        ),
    )
    parser.add_argument(
        "--classifier",
        nargs="+",
        default=cfg.ACTIVE_CLASSIFIERS,
        choices=[*cfg.ACTIVE_CLASSIFIERS, "all"],
        help="Classifier(s) to optimise.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=cfg.N_ROUNDS,
        help="Number of cooperative exchange rounds.",
    )
    parser.add_argument(
        "--ga-gens",
        type=int,
        default=cfg.GA_GENS_PER_ROUND,
        help="GA generations per cooperative round.",
    )
    parser.add_argument(
        "--pso-iters",
        type=int,
        default=cfg.PSO_ITERS_PER_ROUND,
        help="PSO iterations per cooperative round.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        default=False,
        help=(
            "Skip automatic evaluation (confusion matrix, ROC, etc.) after each run. "
            "Useful for quick smoke tests. Run src.evaluate_hybrid separately later."
        ),
    )

    args = parser.parse_args()

    # Expand 'all' shortcut.
    datasets = (
        list(cfg.DATASET_PATHS.keys())
        if "all" in args.dataset
        else args.dataset
    )
    classifiers = (
        cfg.ACTIVE_CLASSIFIERS
        if "all" in args.classifier
        else args.classifier
    )

    # Filter unavailable classifiers.
    if not _XGBOOST_AVAILABLE and "XGBoost" in classifiers:
        logger.warning("XGBoost not installed — skipping XGBoost classifier.")
        classifiers = [c for c in classifiers if c != "XGBoost"]

    # Seed global RNG for reproducibility.
    random.seed(cfg.RANDOM_STATE)
    np.random.seed(cfg.RANDOM_STATE)

    all_results: list[RunResult] = []

    for dataset_name in datasets:
        # Load data once per dataset; share across classifier runs.
        try:
            X, y = load_dataset(dataset_name)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            logger.error("Skipping dataset '%s': %s", dataset_name, exc)
            continue

        for clf_name in classifiers:
            logger.info(
                "\nStarting cooperative run: dataset=%s  classifier=%s",
                dataset_name, clf_name,
            )
            try:
                result = run_cooperative_loop(
                    X=X,
                    y=y,
                    classifier_name=clf_name,
                    dataset_name=dataset_name,
                    n_rounds=args.rounds,
                    ga_gens_per_round=args.ga_gens,
                    pso_iters_per_round=args.pso_iters,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Run failed for %s / %s: %s",
                    dataset_name, clf_name, exc,
                    exc_info=True,
                )
                continue

            print_final_report(result)
            saved_path = save_run_result(result)
            all_results.append(result)

            # Auto-generate full evaluation artefacts unless suppressed.
            if not args.no_eval:
                try:
                    from src.evaluate_hybrid import evaluate_result  # lazy import
                    evaluate_result(saved_path)
                except Exception as eval_exc:  # noqa: BLE001
                    logger.warning(
                        "Evaluation step failed for %s / %s: %s  "
                        "(run  python -m src.evaluate_hybrid --result all  later)",
                        dataset_name, clf_name, eval_exc,
                    )

    # ------------------------------------------------------------------
    # Summary table across all runs
    # ------------------------------------------------------------------
    if all_results:
        print("\n\n" + "=" * 72)
        print("COOPERATIVE GA-PSO  —  FULL ABLATION SUMMARY")
        print("=" * 72)
        header = f"{'Dataset':<15} {'Classifier':<15} {'F1 (best)':>10} {'#Features':>10} {'Time(s)':>8}"
        print(header)
        print("-" * len(header))
        for r in all_results:
            print(
                f"{r.dataset_name:<15} {r.classifier_name:<15} "
                f"{r.best_fitness:>10.4f} {len(r.best_features):>10} "
                f"{r.total_wall_time_s:>8.1f}"
            )
        print("=" * 72)


if __name__ == "__main__":
    main()
