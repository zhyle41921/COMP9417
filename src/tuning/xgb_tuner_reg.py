import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

SEED = 42

def evaluate_xgb_regression(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }

def pick_best_regression(results):
    if not results:
        raise ValueError("No successful XGB regression tuning results.")

    return min(results, key=lambda r: r["val_metrics"]["rmse"])

def clean_best_result(result):
    return {
        "params": result["params"],
        "val_metrics": result["val_metrics"],
    }

def tune_xgb_regression(
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
        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=seed,
            n_jobs=1,
            **params,
        )

        model.fit(X_train, y_train)
        return evaluate_xgb_regression(model, X_val, y_val)

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

    best = pick_best_regression(stage_results)
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

    best = pick_best_regression(stage_results)
    best_params = best["params"]
    results_all.extend(stage_results)

    print("\nBest after N_ESTIMATORS:")
    print(best_params)

    # Stage 3: max depth
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

    best = pick_best_regression(stage_results)
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

    best = pick_best_regression(stage_results)
    best_params = best["params"]
    results_all.extend(stage_results)

    final_best = clean_best_result(pick_best_regression(results_all))

    print("\nFinal best params:")
    print(final_best["params"])

    with open(results_path, "w") as f:
        json.dump(results_all, f, indent=2)

    with open(best_path, "w") as f:
        json.dump(final_best, f, indent=2)

    return final_best, results_all
