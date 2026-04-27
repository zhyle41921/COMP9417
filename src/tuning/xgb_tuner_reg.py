import itertools
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


def tune_xgb_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    results_path,
    best_path,
    seed=SEED,
    base_params=None,
    param_grid=None,
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

    if param_grid is None:
        param_grid = {
            "learning_rate": [0.03, 0.05, 0.1],
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5, 6],
            "reg_alpha": [0.0, 0.1],
            "reg_lambda": [1.0, 5.0],
        }

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    results_all = []

    for combo in itertools.product(*values):
        grid_params = dict(zip(keys, combo))

        params = {
            **base_params,
            **grid_params,
        }

        print(f"[GRID] Testing {params}")

        try:
            model = XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=seed,
                n_jobs=-1,
                tree_method="hist",
                **params,
            )

            model.fit(X_train, y_train)

            metrics = evaluate_xgb_regression(model, X_val, y_val)

            result = {
                "params": params,
                "val_metrics": metrics,
            }

            results_all.append(result)
            print(metrics)

        except Exception as e:
            print(f"Failed: {e}")

    final_best = pick_best_regression(results_all)

    print("\nFinal best params:")
    print(final_best["params"])

    with open(results_path, "w") as f:
        json.dump(results_all, f, indent=2)

    with open(best_path, "w") as f:
        json.dump(final_best, f, indent=2)

    return final_best, results_all