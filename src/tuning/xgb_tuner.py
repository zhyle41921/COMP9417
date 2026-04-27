import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

SEED = 42

def evaluate_xgb(model, X, y):
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
        key=lambda r: r["val_metrics"].get(
            "roc_auc",
            r["val_metrics"]["accuracy"]
        )
    )

def clean_best_result(result):
    return {
        "params": result["params"],
        "val_metrics": result["val_metrics"],
    }

def tune_xgb(
    X_train,
    y_train,
    X_val,
    y_val,
    results_path,
    best_path,
    seed=SEED,
    base_params=None,
    learning_rate_values=None,
    estimator_values=None,
    depth_values=None,
    reg_values=None,
):
    random.seed(seed)
    np.random.seed(seed)

    results_path = Path(results_path)
    best_path = Path(best_path)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    if base_params is None:
        base_params = {
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        }

    if learning_rate_values is None:
        learning_rate_values = [0.03, 0.05, 0.1, 0.2]

    if estimator_values is None:
        estimator_values = [50, 100, 200, 300, 500]

    if depth_values is None:
        depth_values = [3, 4, 5, 6, 8]

    if reg_values is None:
        reg_values = [
            {"reg_alpha": 0.0, "reg_lambda": 0.0},
            {"reg_alpha": 0.0, "reg_lambda": 1.0},
            {"reg_alpha": 0.0, "reg_lambda": 5.0},
            {"reg_alpha": 0.1, "reg_lambda": 1.0},
            {"reg_alpha": 1.0, "reg_lambda": 1.0},
        ]

    results_all = []

    def run_model(params):
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            **params,
        )

        model.fit(X_train, y_train)
        return evaluate_xgb(model, X_val, y_val)

    # Stage 1: learning rate
    stage_results = []

    for learning_rate in learning_rate_values:
        params = {
            **base_params,
            "learning_rate": learning_rate,
            "n_estimators": 200,
            "max_depth": 6,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }

        print(f"[LEARNING_RATE] Testing {params}")

        try:
            metrics = run_model(params)
            result = {
                "stage": "learning_rate",
                "params": params,
                "val_metrics": metrics,
            }
            stage_results.append(result)
            print(metrics)

        except Exception as e:
            print(f"Failed: {e}")

    best = pick_best(stage_results)
    best_params = best["params"]
    results_all.extend(stage_results)

    print("\nBest after LEARNING_RATE:")
    print(best_params)

    # Stage 2: n_estimators
    stage_results = []

    for n_estimators in estimator_values:
        params = {
            **best_params,
            "n_estimators": n_estimators,
        }

        print(f"[N_ESTIMATORS] Testing {params}")

        try:
            metrics = run_model(params)
            result = {
                "stage": "n_estimators",
                "params": params,
                "val_metrics": metrics,
            }
            stage_results.append(result)
            print(metrics)

        except Exception as e:
            print(f"Failed: {e}")

    best = pick_best(stage_results)
    best_params = best["params"]
    results_all.extend(stage_results)

    print("\nBest after N_ESTIMATORS:")
    print(best_params)

    # Stage 3: max_depth
    stage_results = []

    for max_depth in depth_values:
        params = {
            **best_params,
            "max_depth": max_depth,
        }

        print(f"[MAX_DEPTH] Testing {params}")

        try:
            metrics = run_model(params)
            result = {
                "stage": "max_depth",
                "params": params,
                "val_metrics": metrics,
            }
            stage_results.append(result)
            print(metrics)

        except Exception as e:
            print(f"Failed: {e}")

    best = pick_best(stage_results)
    best_params = best["params"]
    results_all.extend(stage_results)

    print("\nBest after MAX_DEPTH:")
    print(best_params)

    # Stage 4: regularisation
    stage_results = []

    for reg in reg_values:
        params = {
            **best_params,
            **reg,
        }

        print(f"[REG] Testing {params}")

        try:
            metrics = run_model(params)
            result = {
                "stage": "regularisation",
                "params": params,
                "val_metrics": metrics,
            }
            stage_results.append(result)
            print(metrics)

        except Exception as e:
            print(f"Failed: {e}")

    best = pick_best(stage_results)
    best_params = best["params"]
    results_all.extend(stage_results)

    final_best = clean_best_result(pick_best(results_all))

    print("\nFinal best params:")
    print(final_best["params"])

    with open(results_path, "w") as f:
        json.dump(results_all, f, indent=2)

    with open(best_path, "w") as f:
        json.dump(final_best, f, indent=2)

    return final_best, results_all
