import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from xrfm import xRFM

SEED = 42

def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred))
    }

    if hasattr(model, "predict_proba"):
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
        raise ValueError("No successful xRFM tuning results.")

    return max(
        results,
        key=lambda r: r["val_metrics"].get(
            "roc_auc",
            r["val_metrics"]["accuracy"]
        )
    )

def tune_xrfm(
    X_train,
    y_train,
    X_val,
    y_val,
    results_path,
    best_path,
    seed=SEED,
    base_params=None,
    max_leaf_size_values=None,
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

    if max_leaf_size_values is None:
        max_leaf_size_values = [256, 512, 1024, 2048]

    results_all = []

    def run_model(params):
        model = xRFM(**params, random_state=seed)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        return evaluate_model(model, X_val, y_val)

    for max_leaf_size in max_leaf_size_values:
        params = {
            **base_params,
            "max_leaf_size": max_leaf_size,
        }

        print(f"[MAX_LEAF_SIZE] Testing {params}")

        try:
            metrics = run_model(params)
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
