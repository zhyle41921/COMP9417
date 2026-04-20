SEED = 42

import os
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import random
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(SEED)
np.random.seed(SEED)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from src.utils.preprocessing import preprocess_data


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


def evaluate_xgb(model, X, y):
    y_pred = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred))
    }

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


def tune_xgb(X_train, y_train, X_val, y_val):
    def run_model(params):
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=1,
            **params,
        )
        model.fit(X_train, y_train)
        return evaluate_xgb(model, X_val, y_val)

    def pick_best(results):
        return max(
            results,
            key=lambda r: r["val_metrics"].get(
                "roc_auc", r["val_metrics"]["accuracy"]
            )
        )

    results_all = []

    base_params = {
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    }

    # Stage 1: learning rate
    learning_rate_values = [0.03, 0.05, 0.1, 0.2]
    stage_results = []

    for val in learning_rate_values:
        params = {
            **base_params,
            "learning_rate": val,
            "n_estimators": 200,
            "max_depth": 6,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }
        print(f"[LEARNING_RATE] Testing {params}")

        try:
            metrics = run_model(params)
            stage_results.append({"params": params, "val_metrics": metrics})
            print(metrics)
        except Exception as e:
            print(f"Failed: {e}")

    best = pick_best(stage_results)
    best_params = best["params"]
    results_all.extend(stage_results)

    print("\nBest after LEARNING_RATE:", best_params)

    # Stage 2: number of estimators / boosting rounds
    estimator_values = [50, 100, 200, 300, 500]
    stage_results = []

    for val in estimator_values:
        params = {**best_params, "n_estimators": val}
        print(f"[N_ESTIMATORS] Testing {params}")

        try:
            metrics = run_model(params)
            stage_results.append({"params": params, "val_metrics": metrics})
            print(metrics)
        except Exception as e:
            print(f"Failed: {e}")

    best = pick_best(stage_results)
    best_params = best["params"]
    results_all.extend(stage_results)

    print("\nBest after N_ESTIMATORS:", best_params)

    # Stage 3: tree complexity
    depth_values = [3, 4, 5, 6, 8]
    stage_results = []

    for val in depth_values:
        params = {**best_params, "max_depth": val}
        print(f"[MAX_DEPTH] Testing {params}")

        try:
            metrics = run_model(params)
            stage_results.append({"params": params, "val_metrics": metrics})
            print(metrics)
        except Exception as e:
            print(f"Failed: {e}")

    best = pick_best(stage_results)
    best_params = best["params"]
    results_all.extend(stage_results)

    print("\nBest after MAX_DEPTH:", best_params)

    # Stage 4: regularisation
    reg_values = [
        {"reg_alpha": 0.0, "reg_lambda": 0.0},
        {"reg_alpha": 0.0, "reg_lambda": 1.0},
        {"reg_alpha": 0.0, "reg_lambda": 5.0},
        {"reg_alpha": 0.1, "reg_lambda": 1.0},
        {"reg_alpha": 1.0, "reg_lambda": 1.0},
    ]
    stage_results = []

    for reg in reg_values:
        params = {**best_params, **reg}
        print(f"[REG] Testing {params}")

        try:
            metrics = run_model(params)
            stage_results.append({"params": params, "val_metrics": metrics})
            print(metrics)
        except Exception as e:
            print(f"Failed: {e}")

    best = pick_best(stage_results)
    best_params = best["params"]
    results_all.extend(stage_results)

    print("\nBest after REG:", best_params)

    print("\nFinal best params:", best_params)

    return best, results_all


def main():
    output_dir = Path(__file__).resolve().parents[2] / "outputs" / "ad"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "xgb_results.json"
    best_path = output_dir / "xgb_best_params.json"

    df = load_ad_data()

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        df,
        target_col="label",
        random_state=SEED,
    )

    print("Shapes:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    best_result, results = tune_xgb(
        X_train, y_train,
        X_val, y_val
    )

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(best_path, "w") as f:
        json.dump(best_result, f, indent=2)

    print("\nBest result:")
    print(json.dumps(best_result, indent=2))


if __name__ == "__main__":
    main()