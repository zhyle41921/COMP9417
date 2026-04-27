import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

SEED = 42

def evaluate_rf(model, X, y):
    y_pred = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred))
    }

    try:
        y_score = np.asarray(model.predict_proba(X))
        if y_score.ndim == 2 and y_score.shape[1] >= 2:
            y_score = y_score[:, 1]
        elif y_score.ndim == 2 and y_score.shape[1] == 1:
            y_score = y_score[:, 0]
        metrics["roc_auc"] = float(roc_auc_score(y, y_score))
    except Exception as e:
        print(f"Could not compute ROC-AUC: {e}")

    return metrics

def pick_best(results):
    if not results:
        raise ValueError("No successful tuning results to choose from.")
    return max(
        results,
        key=lambda r: r["val_metrics"].get("roc_auc", r["val_metrics"]["accuracy"])
    )

def clean_best_result(result):
    return {
        "params": result["params"],
        "val_metrics": result["val_metrics"],
    }

def tune_rf(
    X_train,
    y_train,
    X_val,
    y_val,
    results_path,
    best_path,
    seed=SEED,
    base_params=None,
    n_estimators_values=None,
    max_depth_values=None,
    min_samples_split_values=None,
    max_features_values=None,
):
    random.seed(seed)
    np.random.seed(seed)

    results_path = Path(results_path)
    best_path = Path(best_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    if base_params is None:
        base_params = {}

    if n_estimators_values is None:
        n_estimators_values = [50, 100, 200, 500]

    if max_depth_values is None:
        max_depth_values = [None, 5, 10, 20, 30]

    if min_samples_split_values is None:
        min_samples_split_values = [2, 5, 10]

    if max_features_values is None:
        max_features_values = ["sqrt", "log2", None]

    results_all = []

    def run_model(params):
        model = RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            **params,
        )
        model.fit(X_train, y_train)
        return evaluate_rf(model, X_val, y_val)

    # Stage 1: n_estimators
    stage_results = []
    for n_estimators in n_estimators_values:
        params = {**base_params, "n_estimators": n_estimators}
        print(f"[N_ESTIMATORS] Testing {params}")
        try:
            metrics = run_model(params)
            result = {"stage": "n_estimators", "params": params, "val_metrics": metrics}
            stage_results.append(result)
            print(metrics)
        except Exception as e:
            print(f"Failed: {e}")

    best = pick_best(stage_results)
    best_params = best["params"]
    results_all.extend(stage_results)
    print(f"\nBest after N_ESTIMATORS: {best_params}")

    # Stage 2: max_depth
    stage_results = []
    for max_depth in max_depth_values:
        params = {**best_params, "max_depth": max_depth}
        print(f"[MAX_DEPTH] Testing {params}")
        try:
            metrics = run_model(params)
            result = {"stage": "max_depth", "params": params, "val_metrics": metrics}
            stage_results.append(result)
            print(metrics)
        except Exception as e:
            print(f"Failed: {e}")

    best = pick_best(stage_results)
    best_params = best["params"]
    results_all.extend(stage_results)
    print(f"\nBest after MAX_DEPTH: {best_params}")

    # Stage 3: min_samples_split
    stage_results = []
    for min_split in min_samples_split_values:
        params = {**best_params, "min_samples_split": min_split}
        print(f"[MIN_SAMPLES_SPLIT] Testing {params}")
        try:
            metrics = run_model(params)
            result = {"stage": "min_samples_split", "params": params, "val_metrics": metrics}
            stage_results.append(result)
            print(metrics)
        except Exception as e:
            print(f"Failed: {e}")

    best = pick_best(stage_results)
    best_params = best["params"]
    results_all.extend(stage_results)
    print(f"\nBest after MIN_SAMPLES_SPLIT: {best_params}")

    # Stage 4: max_features
    stage_results = []
    for max_f in max_features_values:
        params = {**best_params, "max_features": max_f}
        print(f"[MAX_FEATURES] Testing {params}")
        try:
            metrics = run_model(params)
            result = {"stage": "max_features", "params": params, "val_metrics": metrics}
            stage_results.append(result)
            print(metrics)
        except Exception as e:
            print(f"Failed: {e}")

    best = pick_best(stage_results)
    best_params = best["params"]
    results_all.extend(stage_results)

    final_best = clean_best_result(pick_best(results_all))
    print(f"\nFinal best params: {final_best['params']}")

    with open(results_path, "w") as f:
        json.dump(results_all, f, indent=2)
    with open(best_path, "w") as f:
        json.dump(final_best, f, indent=2)

    return final_best, results_all
