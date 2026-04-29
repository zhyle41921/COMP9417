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

import json

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from xrfm import xRFM

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.experiment import (
    evaluate_classification,
    evaluate_regression,
    fit_with_time,
    load_best_params,
    metric_row,
    print_shapes,
    save_json,
    to_numpy_splits,
    write_metrics_csv,
)
from src.utils.agop import extract_highest_agop_summary
from experiments.wine_quality.load_data import load_wine_splits

def main():
    output_dir = ROOT / "outputs" / "wine_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_wine_splits()
    X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s = splits
    best_xrfm_params, best_xgb_params, best_rf_params = load_best_params(output_dir)

    X_train_np, X_val_np, X_test_np, y_train_np, y_val_np, y_test_np = to_numpy_splits(splits)

    xrfm_model = xRFM(**best_xrfm_params, n_threads=N_THREADS, random_state=SEED)
    xrfm_model, t = fit_with_time(
        xrfm_model, X_train_np, y_train_np, X_val=X_val_np, y_val=y_val_np
    )
    xrfm_metrics = evaluate_regression(
        xrfm_model, X_test_np, y_test_np, include_full_metrics=True, include_total_time=True
    )
    xrfm_metrics["training_time_seconds"] = t

    agop_summary = extract_highest_agop_summary(
        model=xrfm_model,
        feature_names=list(X_train_df.columns),
        output_dir=output_dir,
        top_k=20,
    )

    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=SEED,
        n_jobs=N_THREADS,
        **best_xgb_params,
    )
    xgb_model, t = fit_with_time(xgb_model, X_train_df, y_train_s)
    xgb_metrics = evaluate_regression(
        xgb_model, X_test_df, y_test_s, include_full_metrics=True, include_total_time=True
    )
    xgb_metrics["training_time_seconds"] = t

    rf_model = RandomForestRegressor(random_state=SEED, n_jobs=N_THREADS, **best_rf_params)
    rf_model, t = fit_with_time(rf_model, X_train_df, y_train_s)
    rf_metrics = evaluate_regression(
        rf_model, X_test_df, y_test_s, include_full_metrics=True, include_total_time=True
    )
    rf_metrics["training_time_seconds"] = t

    results = {
        "xrfm": {
            "params": best_xrfm_params,
            "test_metrics": xrfm_metrics,
            "agop_top_eigenvalue": agop_summary["top_eigenvalue"],
            "agop_index": agop_summary["best_agop_index"],
        },
        "xgboost": {"params": best_xgb_params, "test_metrics": xgb_metrics},
        "random_forest": {"params": best_rf_params, "test_metrics": rf_metrics},
    }
    save_json(output_dir / "test_metrics.json", results)

    fields = ["rmse", "training_time_seconds", "inference_time_seconds"]
    metrics_df, metrics_csv_path = write_metrics_csv([
        metric_row("xrfm", xrfm_metrics, fields),
        metric_row("xgboost", xgb_metrics, fields),
        metric_row("random_forest", rf_metrics, fields),
    ], output_dir)



if __name__ == "__main__":
    main()
