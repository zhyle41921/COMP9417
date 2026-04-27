SEED = 42
N_THREADS = 4

import os

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_THREADS)

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xrfm import xRFM
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.bike_sharing.load_data import load_bike_splits
from src.utils.plotting import plot_rmse_vs_n, plot_training_time_vs_n


SUBSAMPLE_SIZES = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]


def load_best_params():
    output_dir = ROOT / "outputs" / "bike_sharing"

    with open(output_dir / "xrfm_best_params.json", "r") as f:
        xrfm_result = json.load(f)

    with open(output_dir / "xgb_best_params.json", "r") as f:
        xgb_result = json.load(f)

    with open(output_dir / "rf_best_params.json", "r") as f:
        rf_result = json.load(f)

    return xrfm_result["params"], xgb_result["params"], rf_result["params"]


def fit_with_time(model, *fit_args, **fit_kwargs):
    start = time.perf_counter()
    model.fit(*fit_args, **fit_kwargs)
    training_time = time.perf_counter() - start
    return model, float(training_time)


def evaluate_model(model, X, y):
    start = time.perf_counter()
    y_pred = model.predict(X)
    inference_time = time.perf_counter() - start

    mse = mean_squared_error(y, y_pred)

    return {
        "rmse": float(np.sqrt(mse)),
        "inference_time_per_sample_seconds": float(inference_time / len(y)),
    }


def make_metrics_csv(xrfm_metrics, xgb_metrics, rf_metrics, subsample_metrics, output_dir):
    metrics_df = pd.DataFrame([
        xrfm_metrics,
        xgb_metrics,
        rf_metrics,
    ])

    if len(subsample_metrics) > 0:
        metrics_df = pd.concat([metrics_df, pd.DataFrame(subsample_metrics)], ignore_index=True)

    metrics_csv_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    return metrics_df, metrics_csv_path


def run_subsample_experiments(
    X_train_df,
    X_val_df,
    X_test_df,
    y_train_s,
    y_val_s,
    y_test_s,
    best_xrfm_params,
    best_xgb_params,
    best_rf_params,
):
    rng = np.random.default_rng(SEED)
    rows = []

    full_train_size = len(X_train_df)

    X_val_np = np.asarray(X_val_df, dtype=np.float32)
    X_test_np = np.asarray(X_test_df, dtype=np.float32)

    y_val_np = np.asarray(y_val_s, dtype=np.float32)
    y_test_np = np.asarray(y_test_s, dtype=np.float32)

    for n in SUBSAMPLE_SIZES:
        print(f"\nRunning subsample size n={n}")

        sample_idx = rng.choice(full_train_size, size=n, replace=False)

        X_sub_df = X_train_df.iloc[sample_idx]
        y_sub_s = y_train_s.iloc[sample_idx]

        X_sub_np = np.asarray(X_sub_df, dtype=np.float32)
        y_sub_np = np.asarray(y_sub_s, dtype=np.float32)

        # xRFM
        xrfm_model = xRFM(
            **best_xrfm_params,
            n_threads=N_THREADS,
            random_state=SEED,
        )

        xrfm_model, t = fit_with_time(
            xrfm_model,
            X_sub_np,
            y_sub_np,
            X_val=X_val_np,
            y_val=y_val_np,
        )

        m = evaluate_model(xrfm_model, X_test_np, y_test_np)

        rows.append({
            "train_size": n,
            "model": "xrfm",
            "rmse": m["rmse"],
            "training_time_seconds": t,
            "inference_time_per_sample_seconds": m["inference_time_per_sample_seconds"],
        })

        # XGB
        xgb_model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=SEED,
            n_jobs=N_THREADS,
            **best_xgb_params,
        )

        xgb_model, t = fit_with_time(xgb_model, X_sub_df, y_sub_s)
        m = evaluate_model(xgb_model, X_test_df, y_test_s)

        rows.append({
            "train_size": n,
            "model": "xgboost",
            "rmse": m["rmse"],
            "training_time_seconds": t,
            "inference_time_per_sample_seconds": m["inference_time_per_sample_seconds"],
        })

        # RF
        rf_model = RandomForestRegressor(
            random_state=SEED,
            n_jobs=N_THREADS,
            **best_rf_params,
        )

        rf_model, t = fit_with_time(rf_model, X_sub_df, y_sub_s)
        m = evaluate_model(rf_model, X_test_df, y_test_s)

        rows.append({
            "train_size": n,
            "model": "random_forest",
            "rmse": m["rmse"],
            "training_time_seconds": t,
            "inference_time_per_sample_seconds": m["inference_time_per_sample_seconds"],
        })

    return rows


def main():
    output_dir = ROOT / "outputs" / "bike_sharing"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s = load_bike_splits()

    best_xrfm_params, best_xgb_params, best_rf_params = load_best_params()

    X_train_np = np.asarray(X_train_df, dtype=np.float32)
    X_val_np = np.asarray(X_val_df, dtype=np.float32)
    X_test_np = np.asarray(X_test_df, dtype=np.float32)

    y_train_np = np.asarray(y_train_s, dtype=np.float32)
    y_val_np = np.asarray(y_val_s, dtype=np.float32)
    y_test_np = np.asarray(y_test_s, dtype=np.float32)

    # ===== FULL TRAIN =====
    xrfm_model = xRFM(**best_xrfm_params, n_threads=N_THREADS, random_state=SEED)
    xrfm_model, t = fit_with_time(xrfm_model, X_train_np, y_train_np, X_val=X_val_np, y_val=y_val_np)
    m = evaluate_model(xrfm_model, X_test_np, y_test_np)

    xrfm_metrics = {
        "train_size": len(X_train_df),
        "model": "xrfm",
        "rmse": m["rmse"],
        "training_time_seconds": t,
        "inference_time_per_sample_seconds": m["inference_time_per_sample_seconds"],
    }

    xgb_model = XGBRegressor(**best_xgb_params, random_state=SEED, n_jobs=N_THREADS)
    xgb_model, t = fit_with_time(xgb_model, X_train_df, y_train_s)
    m = evaluate_model(xgb_model, X_test_df, y_test_s)

    xgb_metrics = {
        "train_size": len(X_train_df),
        "model": "xgboost",
        "rmse": m["rmse"],
        "training_time_seconds": t,
        "inference_time_per_sample_seconds": m["inference_time_per_sample_seconds"],
    }

    rf_model = RandomForestRegressor(**best_rf_params, random_state=SEED, n_jobs=N_THREADS)
    rf_model, t = fit_with_time(rf_model, X_train_df, y_train_s)
    m = evaluate_model(rf_model, X_test_df, y_test_s)

    rf_metrics = {
        "train_size": len(X_train_df),
        "model": "random_forest",
        "rmse": m["rmse"],
        "training_time_seconds": t,
        "inference_time_per_sample_seconds": m["inference_time_per_sample_seconds"],
    }

    # ===== SUBSAMPLING =====
    subsample_metrics = run_subsample_experiments(
        X_train_df,
        X_val_df,
        X_test_df,
        y_train_s,
        y_val_s,
        y_test_s,
        best_xrfm_params,
        best_xgb_params,
        best_rf_params,
    )

    metrics_df, metrics_csv_path = make_metrics_csv(
        xrfm_metrics,
        xgb_metrics,
        rf_metrics,
        subsample_metrics,
        output_dir,
    )

    print("\nSaved metrics to:", metrics_csv_path)
    print(metrics_df)

    # ===== PLOTTING =====
    plot_rmse_vs_n(metrics_df, output_dir)
    plot_training_time_vs_n(metrics_df, output_dir)


if __name__ == "__main__":
    main()