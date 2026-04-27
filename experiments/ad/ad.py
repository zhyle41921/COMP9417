SEED = 42

import os
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from xrfm import xRFM

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.ad.load_data import load_ad_splits


def load_best_params():
    output_dir = ROOT / "outputs" / "ad"

    with open(output_dir / "xrfm_best_params.json", "r") as f:
        xrfm_result = json.load(f)

    with open(output_dir / "xgb_best_params.json", "r") as f:
        xgb_result = json.load(f)

    with open(output_dir / "rf_best_params.json", "r") as f:
        rf_result = json.load(f)

    return xrfm_result["params"], xgb_result["params"], rf_result["params"]


def to_numpy(X_train, X_val, X_test, y_train, y_val, y_test):
    return (
        np.asarray(X_train, dtype=np.float32),
        np.asarray(X_val, dtype=np.float32),
        np.asarray(X_test, dtype=np.float32),
        np.asarray(y_train, dtype=np.int64),
        np.asarray(y_val, dtype=np.int64),
        np.asarray(y_test, dtype=np.int64),
    )


def fit_with_time(model, *fit_args, **fit_kwargs):
    start = time.perf_counter()
    model.fit(*fit_args, **fit_kwargs)
    training_time = time.perf_counter() - start
    return model, float(training_time)


def evaluate_model(model, X, y):
    start = time.perf_counter()
    y_pred = model.predict(X)
    inference_time = time.perf_counter() - start

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "inference_time_per_sample_seconds": float(inference_time / len(y)),
    }

    if hasattr(model, "predict_proba"):
        y_score = np.asarray(model.predict_proba(X))

        if y_score.ndim == 2 and y_score.shape[1] >= 2:
            y_score = y_score[:, 1]
        elif y_score.ndim == 2:
            y_score = y_score[:, 0]

        metrics["roc_auc"] = float(roc_auc_score(y, y_score))

    return metrics


def make_metrics_csv(xrfm_metrics, xgb_metrics, rf_metrics, output_dir):
    metrics_df = pd.DataFrame([
        {
            "model": "xrfm",
            "accuracy": xrfm_metrics["accuracy"],
            "roc_auc": xrfm_metrics.get("roc_auc", ""),
            "training_time_seconds": xrfm_metrics["training_time_seconds"],
            "inference_time_per_sample_seconds": xrfm_metrics["inference_time_per_sample_seconds"],
        },
        {
            "model": "xgboost",
            "accuracy": xgb_metrics["accuracy"],
            "roc_auc": xgb_metrics.get("roc_auc", ""),
            "training_time_seconds": xgb_metrics["training_time_seconds"],
            "inference_time_per_sample_seconds": xgb_metrics["inference_time_per_sample_seconds"],
        },
        {
            "model": "random_forest",
            "accuracy": rf_metrics["accuracy"],
            "roc_auc": rf_metrics.get("roc_auc", ""),
            "training_time_seconds": rf_metrics["training_time_seconds"],
            "inference_time_per_sample_seconds": rf_metrics["inference_time_per_sample_seconds"],
        },
    ])

    metrics_csv_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    return metrics_df, metrics_csv_path


def main():
    output_dir = ROOT / "outputs" / "ad"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s = load_ad_splits()

    best_xrfm_params, best_xgb_params, best_rf_params = load_best_params()

    X_train_np, X_val_np, X_test_np, y_train_np, y_val_np, y_test_np = to_numpy(
        X_train_df, X_val_df, X_test_df,
        y_train_s, y_val_s, y_test_s,
    )

    # xRFM
    xrfm_model = xRFM(**best_xrfm_params, random_state=SEED)
    xrfm_model, xrfm_training_time = fit_with_time(
        xrfm_model,
        X_train_np, y_train_np,
        X_val=X_val_np, y_val=y_val_np,
    )

    xrfm_metrics = evaluate_model(xrfm_model, X_test_np, y_test_np)
    xrfm_metrics["training_time_seconds"] = xrfm_training_time

    # XGBoost
    from xgboost import XGBClassifier

    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=1,
        **best_xgb_params,
    )

    xgb_model, xgb_training_time = fit_with_time(
        xgb_model,
        X_train_df, y_train_s,
    )

    xgb_metrics = evaluate_model(xgb_model, X_test_df, y_test_s)
    xgb_metrics["training_time_seconds"] = xgb_training_time

    # Random Forest
    from sklearn.ensemble import RandomForestClassifier

    rf_model = RandomForestClassifier(
        random_state=SEED,
        n_jobs=1,
        **best_rf_params,
    )

    rf_model, rf_training_time = fit_with_time(
        rf_model,
        X_train_df, y_train_s,
    )

    rf_metrics = evaluate_model(rf_model, X_test_df, y_test_s)
    rf_metrics["training_time_seconds"] = rf_training_time

    results = {
        "xrfm": {"test_metrics": xrfm_metrics},
        "xgboost": {"test_metrics": xgb_metrics},
        "random_forest": {"test_metrics": rf_metrics},
    }

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    metrics_df, metrics_csv_path = make_metrics_csv(
        xrfm_metrics, xgb_metrics, rf_metrics, output_dir
    )

    print("\nSaved metrics to:", metrics_csv_path)
    print(metrics_df)


if __name__ == "__main__":
    main()