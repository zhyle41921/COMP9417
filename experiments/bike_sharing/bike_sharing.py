SEED = 42
N_THREADS = 4

import os
import sys
from pathlib import Path

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_THREADS)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from xrfm import xRFM

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.bike_sharing.load_data import load_bike_splits
from src.utils.experiment import (
    evaluate_regression,
    fit_with_time,
    load_best_params,
    metric_row,
    to_numpy_splits,
    write_metrics_csv,
)
from src.utils.plotting import plot_rmse_vs_n, plot_training_time_vs_n

SUBSAMPLE_SIZES = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]


def evaluate_three_models(
    X_train_df,
    X_val_df,
    X_test_df,
    y_train_s,
    y_val_s,
    y_test_s,
    best_xrfm_params,
    best_xgb_params,
    best_rf_params,
    train_size,
):
    X_train_np, X_val_np, X_test_np, y_train_np, y_val_np, y_test_np = to_numpy_splits(
        (X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s)
    )

    xrfm_model = xRFM(**best_xrfm_params, n_threads=N_THREADS, random_state=SEED)
    xrfm_model, t = fit_with_time(
        xrfm_model, X_train_np, y_train_np, X_val=X_val_np, y_val=y_val_np
    )
    xrfm_metrics = evaluate_regression(xrfm_model, X_test_np, y_test_np)
    xrfm_metrics["training_time_seconds"] = t

    xgb_model = XGBRegressor(**best_xgb_params, random_state=SEED, n_jobs=N_THREADS)
    xgb_model, t = fit_with_time(xgb_model, X_train_df, y_train_s)
    xgb_metrics = evaluate_regression(xgb_model, X_test_df, y_test_s)
    xgb_metrics["training_time_seconds"] = t

    rf_model = RandomForestRegressor(**best_rf_params, random_state=SEED, n_jobs=N_THREADS)
    rf_model, t = fit_with_time(rf_model, X_train_df, y_train_s)
    rf_metrics = evaluate_regression(rf_model, X_test_df, y_test_s)
    rf_metrics["training_time_seconds"] = t

    fields = ["rmse", "training_time_seconds", "inference_time_per_sample_seconds"]
    extra = {"train_size": train_size}
    return [
        metric_row("xrfm", xrfm_metrics, fields, extra),
        metric_row("xgboost", xgb_metrics, fields, extra),
        metric_row("random_forest", rf_metrics, fields, extra),
    ]


def run_subsample_experiments(splits, best_xrfm_params, best_xgb_params, best_rf_params):
    X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s = splits
    rng = np.random.default_rng(SEED)
    rows = []

    for n in SUBSAMPLE_SIZES:
        print(f"\nRunning subsample size n={n}")
        sample_idx = rng.choice(len(X_train_df), size=n, replace=False)

        rows.extend(evaluate_three_models(
            X_train_df.iloc[sample_idx],
            X_val_df,
            X_test_df,
            y_train_s.iloc[sample_idx],
            y_val_s,
            y_test_s,
            best_xrfm_params,
            best_xgb_params,
            best_rf_params,
            train_size=n,
        ))

    return rows


def main():
    output_dir = ROOT / "outputs" / "bike_sharing"
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_bike_splits()
    X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s = splits
    best_xrfm_params, best_xgb_params, best_rf_params = load_best_params(output_dir)

    rows = evaluate_three_models(
        X_train_df,
        X_val_df,
        X_test_df,
        y_train_s,
        y_val_s,
        y_test_s,
        best_xrfm_params,
        best_xgb_params,
        best_rf_params,
        train_size=len(X_train_df),
    )

    rows.extend(run_subsample_experiments(
        splits, best_xrfm_params, best_xgb_params, best_rf_params
    ))

    metrics_df, metrics_csv_path = write_metrics_csv(rows, output_dir)

    print("\nSaved metrics to:", metrics_csv_path)
    print(metrics_df)

    plot_rmse_vs_n(metrics_df, output_dir)
    plot_training_time_vs_n(metrics_df, output_dir)


if __name__ == "__main__":
    main()
