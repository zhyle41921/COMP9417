import itertools
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

def evaluate_classification(model, X, y):
    y_pred = model.predict(X)
    metrics = {"accuracy": float(accuracy_score(y, y_pred))}

    try:
        y_score = np.asarray(model.predict_proba(X))
        if y_score.ndim == 2 and y_score.shape[1] >= 2:
            y_score = y_score[:, 1]
        elif y_score.ndim == 2 and y_score.shape[1] == 1:
            y_score = y_score[:, 0]
        metrics["roc_auc"] = float(roc_auc_score(y, y_score))
    except Exception:
        pass

    return metrics

def evaluate_regression(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }

def pick_best_classification(results):
    if not results:
        raise ValueError("No successful tuning results to choose from.")
    return max(
        results,
        key=lambda r: r["val_metrics"].get("roc_auc", r["val_metrics"]["accuracy"]),
    )

def pick_best_regression(results):
    if not results:
        raise ValueError("No successful tuning results to choose from.")
    return min(results, key=lambda r: r["val_metrics"]["rmse"])

def run_grid_search(
    X_train,
    y_train,
    X_val,
    y_val,
    results_path,
    best_path,
    seed,
    base_params,
    param_grid,
    make_model,
    evaluate_model,
    pick_best,
    fit_kwargs=None,
):
    random.seed(seed)
    np.random.seed(seed)

    results_path = Path(results_path)
    best_path = Path(best_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    fit_kwargs = fit_kwargs or {}
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    results_all = []

    for combo in itertools.product(*values):
        grid_params = dict(zip(keys, combo))
        params = {**base_params, **grid_params}

        try:
            model = make_model(params, seed)
            model.fit(X_train, y_train, **fit_kwargs)
            metrics = evaluate_model(model, X_val, y_val)
            result = {"params": params, "val_metrics": metrics}
            results_all.append(result)
        except Exception:
            pass

    final_best = pick_best(results_all)

    with open(results_path, "w") as f:
        json.dump(results_all, f, indent=2)

    with open(best_path, "w") as f:
        json.dump(final_best, f, indent=2)

    return final_best, results_all
