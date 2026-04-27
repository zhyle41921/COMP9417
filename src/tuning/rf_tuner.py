import itertools
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
        key=lambda r: r["val_metrics"].get(
            "roc_auc",
            r["val_metrics"]["accuracy"]
        )
    )


def tune_rf(
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
        base_params = {}

    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2", None],
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
            model = RandomForestClassifier(
                random_state=seed,
                n_jobs=-1,
                **params,
            )

            model.fit(X_train, y_train)

            metrics = evaluate_rf(model, X_val, y_val)

            result = {
                "params": params,
                "val_metrics": metrics,
            }

            results_all.append(result)
            print(metrics)

        except Exception as e:
            print(f"Failed: {e}")

    final_best = pick_best(results_all)

    print("\nFinal best params:")
    print(final_best["params"])

    with open(results_path, "w") as f:
        json.dump(results_all, f, indent=2)

    with open(best_path, "w") as f:
        json.dump(final_best, f, indent=2)

    return final_best, results_all