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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xrfm import xRFM
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.preprocessing import preprocess_data
from experiments.wine_quality.load_data import load_wine_splits


def load_best_params():
    output_dir = ROOT / "outputs" / "wine_quality"

    with open(output_dir / "xrfm_best_params.json", "r") as f:
        xrfm_result = json.load(f)

    with open(output_dir / "xgb_best_params.json", "r") as f:
        xgb_result = json.load(f)

    with open(output_dir / "rf_best_params.json", "r") as f:
        rf_result = json.load(f)

    return xrfm_result["params"], xgb_result["params"], rf_result["params"]


def evaluate_model(model, X, y):
    start = time.perf_counter()
    y_pred = model.predict(X)
    inference_time = time.perf_counter() - start

    mse = mean_squared_error(y, y_pred)

    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
        "inference_time_seconds": float(inference_time),
    }


def fit_with_time(model, *fit_args, **fit_kwargs):
    start = time.perf_counter()
    model.fit(*fit_args, **fit_kwargs)
    training_time = time.perf_counter() - start
    return model, float(training_time)


def to_numpy_agop(agop):
    if hasattr(agop, "detach"):
        agop = agop.detach().cpu().numpy()
    else:
        agop = np.asarray(agop)

    if agop.ndim == 1:
        agop = np.diag(agop)

    return agop


def extract_highest_agop_summary(model, feature_names, output_dir, top_k=20):
    agops = model.collect_best_agops()

    best_agop = None
    best_score = -np.inf
    best_index = None

    for i, agop in enumerate(agops):
        agop = to_numpy_agop(agop)

        eigenvalues = np.linalg.eigvalsh(agop)
        score = float(np.max(eigenvalues))

        if score > best_score:
            best_score = score
            best_agop = agop
            best_index = i

    if best_agop is None:
        raise ValueError("No AGOP matrices found.")

    if best_agop.shape[0] != len(feature_names):
        raise ValueError(
            f"AGOP dimension {best_agop.shape[0]} does not match "
            f"{len(feature_names)} feature names."
        )

    agop_df = pd.DataFrame(
        best_agop,
        index=feature_names,
        columns=feature_names,
    )

    agop_path = output_dir / "xrfm_best_agop.csv"
    agop_df.to_csv(agop_path)

    print("\nSaved full AGOP matrix to:", agop_path)

    diag = np.diag(best_agop)

    diag_df = pd.DataFrame({
        "feature": feature_names,
        "agop_diagonal": diag,
        "abs_agop_diagonal": np.abs(diag),
    }).sort_values("abs_agop_diagonal", ascending=False)

    eigenvalues, eigenvectors = np.linalg.eigh(best_agop)
    top_idx = int(np.argmax(eigenvalues))

    top_eigenvalue = float(eigenvalues[top_idx])
    top_eigenvector = eigenvectors[:, top_idx]

    eigen_df = pd.DataFrame({
        "feature": feature_names,
        "top_eigenvector_loading": top_eigenvector,
        "abs_loading": np.abs(top_eigenvector),
    }).sort_values("abs_loading", ascending=False)

    diag_path = output_dir / "xrfm_highest_agop_diagonal.csv"
    eigen_path = output_dir / "xrfm_highest_agop_top_eigenvector_loadings.csv"

    diag_df.to_csv(diag_path, index=False)
    eigen_df.to_csv(eigen_path, index=False)

    return {
        "top_diag_df": diag_df.head(top_k),
        "top_eigen_df": eigen_df.head(top_k),
        "top_eigenvalue": top_eigenvalue,
        "best_agop_index": best_index,
        "diag_path": str(diag_path),
        "eigen_path": str(eigen_path),
    }


def make_metrics_csv(xrfm_metrics, xgb_metrics, rf_metrics, output_dir):
    metrics_df = pd.DataFrame([
        {
            "model": "xrfm",
            "rmse": xrfm_metrics["rmse"],
            "training_time_seconds": xrfm_metrics["training_time_seconds"],
            "inference_time_seconds": xrfm_metrics["inference_time_seconds"],
        },
        {
            "model": "xgboost",
            "rmse": xgb_metrics["rmse"],
            "training_time_seconds": xgb_metrics["training_time_seconds"],
            "inference_time_seconds": xgb_metrics["inference_time_seconds"],
        },
        {
            "model": "random_forest",
            "rmse": rf_metrics["rmse"],
            "training_time_seconds": rf_metrics["training_time_seconds"],
            "inference_time_seconds": rf_metrics["inference_time_seconds"],
        },
    ])

    metrics_csv_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    return metrics_df, metrics_csv_path


def main():
    output_dir = ROOT / "outputs" / "wine_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s = load_wine_splits()

    best_xrfm_params, best_xgb_params, best_rf_params = load_best_params()

    X_train_np = np.asarray(X_train_df, dtype=np.float32)
    X_val_np = np.asarray(X_val_df, dtype=np.float32)
    X_test_np = np.asarray(X_test_df, dtype=np.float32)

    y_train_np = np.asarray(y_train_s, dtype=np.float32)
    y_val_np = np.asarray(y_val_s, dtype=np.float32)
    y_test_np = np.asarray(y_test_s, dtype=np.float32)

    xrfm_model = xRFM(
        **best_xrfm_params,
        n_threads=N_THREADS,
        random_state=SEED,
    )

    xrfm_model, xrfm_training_time = fit_with_time(
        xrfm_model,
        X_train_np,
        y_train_np,
        X_val=X_val_np,
        y_val=y_val_np,
    )

    xrfm_metrics = evaluate_model(xrfm_model, X_test_np, y_test_np)
    xrfm_metrics["training_time_seconds"] = xrfm_training_time

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

    xgb_model, xgb_training_time = fit_with_time(
        xgb_model,
        X_train_df,
        y_train_s,
    )

    xgb_metrics = evaluate_model(xgb_model, X_test_df, y_test_s)
    xgb_metrics["training_time_seconds"] = xgb_training_time

    rf_model = RandomForestRegressor(
        random_state=SEED,
        n_jobs=N_THREADS,
        **best_rf_params,
    )

    rf_model, rf_training_time = fit_with_time(
        rf_model,
        X_train_df,
        y_train_s,
    )

    rf_metrics = evaluate_model(rf_model, X_test_df, y_test_s)
    rf_metrics["training_time_seconds"] = rf_training_time

    results = {
        "xrfm": {
            "params": best_xrfm_params,
            "test_metrics": xrfm_metrics,
        },
        "xgboost": {
            "params": best_xgb_params,
            "test_metrics": xgb_metrics,
        },
        "random_forest": {
            "params": best_rf_params,
            "test_metrics": rf_metrics,
        },
    }

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    metrics_df, metrics_csv_path = make_metrics_csv(
        xrfm_metrics=xrfm_metrics,
        xgb_metrics=xgb_metrics,
        rf_metrics=rf_metrics,
        output_dir=output_dir,
    )
    print("\nTest metrics:")
    print(json.dumps(results, indent=2))

    print("\nSaved combined metrics to:", metrics_csv_path)
    print(metrics_df)


if __name__ == "__main__":
    main()