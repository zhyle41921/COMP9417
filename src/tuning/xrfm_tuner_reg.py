import itertools
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xrfm import xRFM

SEED = 42


def evaluate_regression_model(model, X, y):
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
        raise ValueError("No successful xRFM regression tuning results.")

    return min(results, key=lambda r: r["val_metrics"]["rmse"])


def tune_xrfm_regression(
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
            "verbose": False,
        }

    if param_grid is None:
        param_grid = {
            "max_leaf_size": [256, 512, 1024, 2048],
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
            model = xRFM(**params, random_state=seed)
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

            metrics = evaluate_regression_model(model, X_val, y_val)

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