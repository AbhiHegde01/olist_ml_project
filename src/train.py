"""
Train, compare, tune and evaluate models for both tasks (late delivery, satisfaction).

Outputs
    models/late.joblib, models/satisfaction.joblib   (pipeline + threshold + metadata)
    models/metrics.json                              (everything the dashboard shows)

Run:  python src/train.py            full run: compare 6 models + hyper-parameter search (~15-20 min)
      python src/train.py --fast     re-use the tuned XGBoost parameters saved in metrics.json,
                                     skip the search, lighter Random Forest (~2-3 min)
      python src/train.py late       only one task (combine with --fast)
"""
from __future__ import annotations

import os

# Cap native thread pools BEFORE numpy is imported. By default OpenBLAS starts one thread
# (with its own memory buffer) per CPU core; on a 22-thread laptop running several Python
# processes this exhausted memory and killed training silently. Override with env vars.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")
N_JOBS = int(os.environ.get("TRAIN_N_JOBS", "8"))   # cores for Random Forest / XGBoost

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import TASKS, build_preprocessor, input_defaults, modeling_frame  # noqa: E402

MODELS_DIR = ROOT / "models"
SPLIT_DATE = "2018-04-01"   # train: 2017-01 .. 2018-03 | test: 2018-04 .. 2018-08
SEED = 42
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1",
           "precision": "precision", "recall": "recall"}


FAST = "--fast" in sys.argv


def candidate_models(task: str) -> dict[str, Pipeline]:
    """Each model is a full pipeline: preprocessing + classifier."""
    return {
        "Naive Bayes (Gaussian)": Pipeline([("prep", build_preprocessor(task)),
                                            ("clf", GaussianNB())]),
        "Logistic Regression": Pipeline([("prep", build_preprocessor(task)),
                                         ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                                                    class_weight="balanced"))]),
        "Decision Tree": Pipeline([("prep", build_preprocessor(task, scale=False)),
                                   ("clf", DecisionTreeClassifier(max_depth=8, min_samples_leaf=50,
                                                                  class_weight="balanced",
                                                                  random_state=SEED))]),
        "Random Forest": Pipeline([("prep", build_preprocessor(task, scale=False)),
                                   ("clf", RandomForestClassifier(n_estimators=80 if FAST else 200, max_depth=14,
                                                                  min_samples_leaf=5,
                                                                  class_weight="balanced_subsample",
                                                                  n_jobs=N_JOBS, random_state=SEED))]),
        "XGBoost": Pipeline([("prep", build_preprocessor(task, scale=False)),
                             ("clf", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                                                   subsample=0.8, colsample_bytree=0.8,
                                                   tree_method="hist", eval_metric="aucpr",
                                                   n_jobs=N_JOBS, random_state=SEED))]),
    }


XGB_SEARCH = {
    "clf__n_estimators": [300, 600],
    "clf__learning_rate": [0.02, 0.05, 0.1],
    "clf__max_depth": [4, 6, 8],
    "clf__min_child_weight": [1, 5, 10],
    "clf__subsample": [0.7, 0.85, 1.0],
    "clf__colsample_bytree": [0.6, 0.8, 1.0],
    "clf__reg_lambda": [1, 5, 10],
}


def test_metrics(y, proba, thr=0.5) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba),
    }


def best_f1_threshold(y, proba) -> float:
    p, r, t = precision_recall_curve(y, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(t[np.nanargmax(f1[:-1])])


def thin(a, n=200):
    """Downsample curve points for a compact json."""
    a = np.asarray(a)
    idx = np.linspace(0, len(a) - 1, min(n, len(a))).astype(int)
    return a[idx].round(4).tolist()


def run_task(task: str, master: pd.DataFrame, prev_params: dict | None = None) -> dict:
    cfg = TASKS[task]
    print(f"\n=== {cfg['title']} ===")
    X, y, ts = modeling_frame(master, task)
    tr = ts < SPLIT_DATE
    X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[~tr], y[~tr]
    print(f"train {len(X_tr):,} (pos {y_tr.mean():.3f}) | test {len(X_te):,} (pos {y_te.mean():.3f})")

    # 1) cross-validated comparison on the training period
    cv_rows, test_rows, fitted = [], [], {}
    for name, pipe in candidate_models(task).items():
        t0 = time.time()
        cv = cross_validate(pipe, X_tr, y_tr, cv=CV, scoring=SCORING, n_jobs=1)
        row = {"model": name}
        for k in SCORING:
            row[k] = float(cv[f"test_{k}"].mean())
            row[f"{k}_std"] = float(cv[f"test_{k}"].std())
        cv_rows.append(row)
        pipe.fit(X_tr, y_tr)
        fitted[name] = pipe
        test_rows.append({"model": name, **test_metrics(y_te, pipe.predict_proba(X_te)[:, 1])})
        print(f"  {name:<24} CV PR-AUC {row['pr_auc']:.3f}  ROC-AUC {row['roc_auc']:.3f}  ({time.time()-t0:.0f}s)")

    # 2) hyper-parameter tuning of XGBoost (--fast re-uses the parameters found by the last full run)
    t0 = time.time()
    if FAST and prev_params:
        print("  --fast: re-using tuned XGBoost parameters from metrics.json")
        best_params = prev_params
        tuned = clone(candidate_models(task)["XGBoost"]).set_params(**best_params).fit(X_tr, y_tr)
    else:
        print("  tuning XGBoost (RandomizedSearchCV, 8 candidates x 3 folds) ...")
        search = RandomizedSearchCV(candidate_models(task)["XGBoost"], XGB_SEARCH, n_iter=8,
                                    scoring="average_precision", cv=CV, random_state=SEED, n_jobs=1)
        search.fit(X_tr, y_tr)
        tuned = search.best_estimator_
        best_params = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in search.best_params_.items()}
    cv = cross_validate(tuned, X_tr, y_tr, cv=CV, scoring=SCORING)
    row = {"model": "XGBoost (tuned)"}
    for k in SCORING:
        row[k] = float(cv[f"test_{k}"].mean())
        row[f"{k}_std"] = float(cv[f"test_{k}"].std())
    cv_rows.append(row)
    fitted["XGBoost (tuned)"] = tuned
    test_rows.append({"model": "XGBoost (tuned)", **test_metrics(y_te, tuned.predict_proba(X_te)[:, 1])})
    print(f"  XGBoost (tuned)          CV PR-AUC {row['pr_auc']:.3f}  ({time.time()-t0:.0f}s)  params {best_params}")

    # 3) choose best model by CV PR-AUC (never by test score)
    best_name = max(cv_rows, key=lambda r: r["pr_auc"])["model"]
    best = fitted[best_name]
    print(f"  best model: {best_name}")

    # 4) decision threshold from out-of-fold predictions on TRAIN
    oof = cross_val_predict(clone(best), X_tr, y_tr, cv=CV, method="predict_proba")[:, 1]
    thr = best_f1_threshold(y_tr, oof)

    # 5) final evaluation on the untouched test period
    proba = best.predict_proba(X_te)[:, 1]
    final = test_metrics(y_te, proba, thr)
    pred = (proba >= thr).astype(int)
    fpr, tpr, _ = roc_curve(y_te, proba)
    prec, rec, _ = precision_recall_curve(y_te, proba)
    frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=10, strategy="quantile")

    print("  permutation importance ...")
    sample = X_te.sample(min(6000, len(X_te)), random_state=SEED)
    pi = permutation_importance(best, sample, y_te.loc[sample.index], scoring="average_precision",
                                n_repeats=2 if FAST else 5, random_state=SEED, n_jobs=1)
    imp = (pd.DataFrame({"feature": sample.columns, "importance": pi.importances_mean,
                         "std": pi.importances_std})
           .sort_values("importance", ascending=False))

    print(f"  TEST @thr={thr:.3f}: " + ", ".join(f"{k} {v:.3f}" for k, v in final.items()))

    # 6) save
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({
        "task": task, "model_name": best_name, "pipeline": best, "threshold": thr,
        "features": list(X.columns), "numeric": cfg["numeric"], "categorical": cfg["categorical"],
        "defaults": input_defaults(X_tr),
        "categories": {c: sorted(X_tr[c].dropna().unique().tolist()) for c in cfg["categorical"]},
        "base_rate": float(y_tr.mean()),
    }, MODELS_DIR / f"{task}.joblib", compress=3)

    return {
        "task": task, "title": cfg["title"], "target": cfg["target"],
        "positive_label": cfg["positive_label"],
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "train_period": f"{ts[tr].min():%Y-%m-%d} to {ts[tr].max():%Y-%m-%d}",
        "test_period": f"{ts[~tr].min():%Y-%m-%d} to {ts[~tr].max():%Y-%m-%d}",
        "pos_rate_train": float(y_tr.mean()), "pos_rate_test": float(y_te.mean()),
        "features": list(X.columns),
        "cv": cv_rows, "test_all_models_at_0.5": test_rows,
        "best_model": best_name,
        "best_params": best_params,
        "threshold": thr, "test": final,
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "roc_curve": {"fpr": thin(fpr), "tpr": thin(tpr)},
        "pr_curve": {"precision": thin(prec), "recall": thin(rec)},
        "calibration": {"mean_predicted": mean_pred.round(4).tolist(),
                        "fraction_positive": frac_pos.round(4).tolist()},
        "importance": imp.round(5).to_dict(orient="records"),
    }


def main():
    master = pd.read_parquet(ROOT / "data" / "processed" / "master_orders.parquet")
    tasks = [a for a in sys.argv[1:] if not a.startswith("--")] or list(TASKS)
    t_start = time.time()
    path = MODELS_DIR / "metrics.json"
    metrics = json.loads(path.read_text()) if path.exists() else {}
    for task in tasks:
        prev = metrics.get(task, {}).get("best_params")
        if FAST and not prev:
            print(f"  (no saved parameters for '{task}' yet - running the full search once)")
        metrics[task] = run_task(task, master, prev)
        path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved models and {path}  (total {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
