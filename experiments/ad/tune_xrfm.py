SEED = 42

import random

import sys
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from sklearn.metrics import accuracy_score, roc_auc_score
from src.utils.preprocessing import preprocess_data
from xrfm import xRFM

random.seed(SEED)
np.random.seed(SEED)


def load_ad_data():
    n_features = 1558
    col_names = [f"x{i}" for i in range(n_features)] + ["label"]

    data_path = Path(__file__).resolve().parents[2] / "experiments" / "ad" / "ad.data"

    df = pd.read_csv(
        data_path,
        header=None,
        names=col_names,
        na_values="?",
        skipinitialspace=True,
        low_memory=False,
    )

    df["label"] = df["label"].str.strip()
    df["label"] = df["label"].map({"nonad.": 0, "ad.": 1})

    if df["label"].isna().any():
        raise ValueError("Found unmapped labels in ad dataset.")

    return df


def to_numpy(X_train, X_val, X_test, y_train, y_val, y_test):
    return (
        np.asarray(X_train, dtype=np.float32),
        np.asarray(X_val, dtype=np.float32),
        np.asarray(X_test, dtype=np.float32),
        np.asarray(y_train, dtype=np.int64),
        np.asarray(y_val, dtype=np.int64),
        np.asarray(y_test, dtype=np.int64),
    )


def evaluate_xrfm(model, X, y):
    y_pred = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred))
    }

    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X)
            y_score = np.asarray(y_score)
            if y_score.ndim == 2 and y_score.shape[1] >= 2:
                y_score = y_score[:, 1]
            elif y_score.ndim == 2 and y_score.shape[1] == 1:
                y_score = y_score[:, 0]
            metrics["roc_auc"] = float(roc_auc_score(y, y_score))
        except Exception as e:
            print(f"Could not compute ROC-AUC: {e}")

    return metrics


def tune_xrfm(X_train, y_train, X_val, y_val):
    def run_model(params):
        model = xRFM(**params, random_state=SEED)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        return evaluate_xrfm(model, X_val, y_val)

    def pick_best(results):
        return max(
            results,
            key=lambda r: r["val_metrics"].get(
                "roc_auc", r["val_metrics"]["accuracy"]
            )
        )

    results_all = []

    base_params = {
        "max_leaf_size": 512,
        "verbose": False,
    }

    agop_values = [0.25, 0.5, 1.0, 2.0, 4.0]
    stage_results = []

    for val in agop_values:
        params = {**base_params, "agop_power": val, "iters": 3}
        print(f"[AGOP] Testing {params}")

        try:
            metrics = run_model(params)
            stage_results.append({"params": params, "val_metrics": metrics})
            print(metrics)
        except Exception as e:
            print(f"Failed: {e}")

    best = pick_best(stage_results)
    best_params = best["params"]
    results_all.extend(stage_results)

    print("\nBest after AGOP:", best_params)

    iter_values = [1, 2, 3, 4, 5]
    stage_results = []

    for val in iter_values:
        params = {**best_params, "iters": val}
        print(f"[ITERS] Testing {params}")

        try:
            metrics = run_model(params)
            stage_results.append({"params": params, "val_metrics": metrics})
            print(metrics)
        except Exception as e:
            print(f"Failed: {e}")

    best = pick_best(stage_results)
    best_params = best["params"]
    results_all.extend(stage_results)

    print("\nBest after ITERS:", best_params)

    best = pick_best(stage_results)
    best_params = best["params"]
    results_all.extend(stage_results)

    print("\nFinal best params:", best_params)

    return best, results_all

def main():
    output_dir = Path(__file__).resolve().parents[2] / "outputs" / "ad"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "xrfm_results.json"
    best_path = output_dir / "xrfm_best_params.json"

    df = load_ad_data()

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        df,
        target_col="label"
    )

    X_train, X_val, X_test, y_train, y_val, y_test = to_numpy(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    print("Shapes:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    agop_values = [0.25, 0.5, 1.0, 2.0]

    best_result, results = tune_xrfm(
        X_train, y_train,
        X_val, y_val
    )

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    best_result = max(
        results,
        key=lambda r: r["val_metrics"].get("roc_auc", r["val_metrics"]["accuracy"])
    )

    with open(best_path, "w") as f:
        json.dump(best_result, f, indent=2)

    print("\nBest result:")
    print(json.dumps(best_result, indent=2))


if __name__ == "__main__":
    main()