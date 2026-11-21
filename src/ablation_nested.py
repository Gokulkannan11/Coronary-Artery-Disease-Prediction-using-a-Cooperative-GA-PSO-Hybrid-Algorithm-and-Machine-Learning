"""Nested-CV ablation: does the cooperative GA-PSO exchange add anything?

Compares five variants of SVM-based CAD prediction under the SAME outer folds
and the SAME fitness-evaluation budget:

  baseline     untuned RBF-SVM, all 13 features
  ga_only      GA feature selection, default SVM hyperparameters
  pso_only     PSO hyperparameter tuning, all 13 features
  sequential   GA (half budget) then PSO (half budget), one pass, no feedback
  cooperative  R rounds of GA <-> PSO with two-way exchange (proposed)

Outer loop: 5-fold stratified CV x 3 seeds (15 paired test folds).
Inner loop: every search runs only on the outer-training part, scored by
5-fold CV F1 minus 0.005 per feature. The outer test fold is never seen
during search, so the outer metrics are unbiased estimates.

Usage:  python -m src.ablation_nested
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Heart_disease_cleveland_new.csv"
TAG = os.environ.get("ABL_TAG", "")
OUT = ROOT / "results" / ("ablation" + TAG)

SEEDS = [42, 7, 2026]
OUTER_K, INNER_K = 5, 5
LAMBDA = float(os.environ.get("ABL_LAMBDA", "0.005"))
INNER_REPEATS = int(os.environ.get("ABL_REPEATS", "1"))
MIN_FEAT = 2
# GA / PSO operators (same values as src/config.py)
POP, CXPB, MUTPB, INDPB, TOURN = 50, 0.7, 0.2, 0.1, 3
SWARM, C1, C2, W = 30, 2.0, 1.5, 0.4
LO, HI = np.array([-3.0, -4.0]), np.array([3.0, 0.0])  # log10 C, log10 gamma
# Equal budget (fitness evaluations) for every searched variant ~ 1500
ROUNDS, GA_GEN_R, PSO_IT_R = 3, 5, 8          # cooperative: 3 x (250 + 240)
GA_GEN_ONLY, PSO_IT_ONLY = 29, 49             # ~1500 each
GA_GEN_SEQ, PSO_IT_SEQ = 14, 24               # ~750 + ~750
VARIANTS = ["baseline", "ga_only", "pso_only", "sequential", "cooperative"]


class Fitness:
    def __init__(self, X, y, seed):
        self.X, self.y = X, y
        self.cv = RepeatedStratifiedKFold(n_splits=INNER_K, n_repeats=INNER_REPEATS, random_state=seed)
        self.cache, self.evals = {}, 0

    def __call__(self, mask, theta):
        mask = tuple(int(b) for b in mask)
        key = (mask, round(float(theta[0]), 4), round(float(theta[1]), 4))
        if key in self.cache:
            return self.cache[key]
        self.evals += 1
        k = sum(mask)
        if k < MIN_FEAT:
            val = 0.0
        else:
            cols = np.flatnonzero(mask)
            model = make_pipeline(StandardScaler(), SVC(C=10 ** theta[0], gamma=10 ** theta[1]))
            f1 = cross_val_score(model, self.X[:, cols], self.y, cv=self.cv, scoring="f1").mean()
            val = max(0.0, f1 - LAMBDA * k)
        self.cache[key] = val
        return val


def default_theta(n_feat):
    # sklearn defaults: C=1, gamma='scale' ~ 1/n_features on standardised data
    return np.array([0.0, np.log10(1.0 / n_feat)])


def ga(fit, theta, gens, rng, pop=None):
    n = 13
    if pop is None:
        pop = rng.integers(0, 2, size=(POP, n))
    scores = np.array([fit(ind, theta) for ind in pop])
    for _ in range(gens):
        # tournament selection
        idx = [max(rng.choice(POP, TOURN, replace=False), key=lambda i: scores[i]) for _ in range(POP)]
        off = pop[idx].copy()
        for i in range(0, POP - 1, 2):
            if rng.random() < CXPB:
                a, b = sorted(rng.choice(np.arange(1, n), 2, replace=False))
                off[i, a:b], off[i + 1, a:b] = off[i + 1, a:b].copy(), off[i, a:b].copy()
        for i in range(POP):
            if rng.random() < MUTPB:
                flip = rng.random(n) < INDPB
                off[i, flip] = 1 - off[i, flip]
        off_scores = np.array([fit(ind, theta) for ind in off])
        best = int(np.argmax(scores))
        worst = int(np.argmin(off_scores))
        off[worst], off_scores[worst] = pop[best], scores[best]   # elitism
        pop, scores = off, off_scores
    b = int(np.argmax(scores))
    return pop[b].copy(), float(scores[b]), pop


def pso(fit, mask, iters, rng, start=None):
    d = 2
    if start is None:
        x = rng.uniform(LO, HI, size=(SWARM, d))
    else:
        x = np.clip(start + rng.uniform(-0.2, 0.2, size=(SWARM, d)) * (HI - LO), LO, HI)
        x[0] = start
    v = np.zeros_like(x)
    pb = x.copy()
    pbs = np.array([fit(mask, p) for p in x])
    g = pb[int(np.argmax(pbs))].copy()
    gs = float(pbs.max())
    for _ in range(iters):
        r1, r2 = rng.random((SWARM, d)), rng.random((SWARM, d))
        v = W * v + C1 * r1 * (pb - x) + C2 * r2 * (g - x)
        x = np.clip(x + v, LO, HI)
        s = np.array([fit(mask, p) for p in x])
        better = s > pbs
        pb[better], pbs[better] = x[better], s[better]
        if pbs.max() > gs:
            gs = float(pbs.max())
            g = pb[int(np.argmax(pbs))].copy()
    return g, gs


def search(variant, X, y, seed):
    rng = np.random.default_rng(seed)
    fit = Fitness(X, y, seed)
    all_mask = np.ones(13, dtype=int)
    trace = []
    if variant == "baseline":
        return all_mask, default_theta(13), fit, trace
    if variant == "ga_only":
        th = default_theta(13)
        m, _, _ = ga(fit, th, GA_GEN_ONLY, rng)
        return m, th, fit, trace
    if variant == "pso_only":
        th, _ = pso(fit, all_mask, PSO_IT_ONLY, rng)
        return all_mask, th, fit, trace
    if variant == "sequential":
        m, _, _ = ga(fit, default_theta(13), GA_GEN_SEQ, rng)
        th, _ = pso(fit, m, PSO_IT_SEQ, rng)
        return m, th, fit, trace
    # cooperative
    th, m, best, pop, start = default_theta(13), all_mask, -1.0, None, None
    for r in range(ROUNDS):
        gm, gs, pop = ga(fit, th, GA_GEN_R, rng, pop)
        if gs > best:
            m, best = gm, gs
        gth, ps = pso(fit, m, PSO_IT_R, rng, start)
        if ps > best:
            th, best = gth, ps
        start = th
        trace.append({"round": r + 1, "ga": gs, "pso": ps, "best": best, "k": int(m.sum())})
    return m, th, fit, trace


def run_fold(seed, fold, tr, te, X, y, names):
    rows = []
    for variant in VARIANTS:
        t0 = time.time()
        m, th, fit, trace = search(variant, X[tr], y[tr], seed * 100 + fold)
        cols = np.flatnonzero(m)
        if variant == "baseline":
            model = make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))
        else:
            model = make_pipeline(StandardScaler(), SVC(C=10 ** th[0], gamma=10 ** th[1],
                                                        probability=True, random_state=0))
        model.fit(X[tr][:, cols], y[tr])
        pred = model.predict(X[te][:, cols])
        prob = model.predict_proba(X[te][:, cols])[:, 1]
        rows.append({
            "seed": seed, "fold": fold, "variant": variant,
            "acc": accuracy_score(y[te], pred), "f1": f1_score(y[te], pred),
            "prec": precision_score(y[te], pred), "rec": recall_score(y[te], pred),
            "auc": roc_auc_score(y[te], prob), "mcc": matthews_corrcoef(y[te], pred),
            "n_feat": int(len(cols)), "C": float(10 ** th[0]), "gamma": float(10 ** th[1]),
            "features": [names[c] for c in cols], "evals": fit.evals,
            "sec": time.time() - t0, "trace": trace,
        })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, comment="#")
    names = [c for c in df.columns if c != "target"]
    X, y = df[names].to_numpy(float), df["target"].to_numpy(int)
    jobs = []
    for seed in SEEDS:
        skf = StratifiedKFold(OUTER_K, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            jobs.append((seed, fold, tr, te))
    t0 = time.time()
    res = Parallel(n_jobs=-1, verbose=5)(delayed(run_fold)(s, f, tr, te, X, y, names)
                                         for s, f, tr, te in jobs)
    rows = [r for fold_rows in res for r in fold_rows]
    (OUT / "ablation_raw.json").write_text(json.dumps(rows, indent=1))
    res_df = pd.DataFrame(rows)
    metrics = ["acc", "f1", "prec", "rec", "auc", "mcc", "n_feat", "evals", "sec"]
    summ = res_df.groupby("variant")[metrics].agg(["mean", "std"]).reindex(VARIANTS)
    summ.to_csv(OUT / "ablation_summary.csv")
    print(summ.round(3).to_string())
    tests = []
    coop = res_df[res_df.variant == "cooperative"].sort_values(["seed", "fold"])
    for v in VARIANTS[:-1]:
        other = res_df[res_df.variant == v].sort_values(["seed", "fold"])
        for met in ["acc", "f1", "mcc"]:
            a, b = coop[met].to_numpy(), other[met].to_numpy()
            d = a - b
            p = wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
            tests.append({"vs": v, "metric": met, "mean_diff": d.mean(),
                          "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
                          "losses": int((d < 0).sum()), "p_wilcoxon": p})
    tdf = pd.DataFrame(tests)
    tdf.to_csv(OUT / "ablation_tests.csv", index=False)
    print(tdf.round(4).to_string())
    feat = pd.Series([f for r in rows if r["variant"] == "cooperative" for f in r["features"]])
    feat.value_counts().to_csv(OUT / "coop_feature_frequency.csv")
    print(feat.value_counts())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
