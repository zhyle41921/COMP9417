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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xrfm import xRFM
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.bike_sharing.load_data import load_bike_splits

def load_best_params():
    output_dir = ROOT / "outputs" / "bike_sharing"

    with open(output_dir / "xrfm_best_params.json", "r") as f:
        xrfm_result = json.load(f)

    with open(output_dir / "xgb_best_params.json", "r") as f:
        xgb_result = json.load(f)

    return xrfm_result["params"], xgb_result["params"]

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }

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

    agop_df = pd.DataFrame(best_agop, index=feature_names, columns=feature_names)
    agop_path = output_dir / "xrfm_best_agop.csv"
    agop_df.to_csv(agop_path)

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
    }

def make_metrics_csv(xrfm_metrics, xgb_metrics, output_dir):
    metrics_df = pd.DataFrame([
        {
            "model": "xrfm",
            "mse": xrfm_metrics["mse"],
            "rmse": xrfm_metrics["rmse"],
            "mae": xrfm_metrics["mae"],
            "r2": xrfm_metrics["r2"],
        },
        {
            "model": "xgboost",
            "mse": xgb_metrics["mse"],
            "rmse": xgb_metrics["rmse"],
            "mae": xgb_metrics["mae"],
            "r2": xgb_metrics["r2"],
        },
    ])

    metrics_csv_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    return metrics_df, metrics_csv_path

def main():
    output_dir = ROOT / "outputs" / "bike_sharing"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s = load_bike_splits()

    best_xrfm_params, best_xgb_params = load_best_params()

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
    xrfm_model.fit(X_train_np, y_train_np, X_val=X_val_np, y_val=y_val_np)
    xrfm_metrics = evaluate_model(xrfm_model, X_test_np, y_test_np)

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
    xgb_model.fit(X_train_df, y_train_s)
    xgb_metrics = evaluate_model(xgb_model, X_test_df, y_test_s)

    results = {
        "xrfm": {
            "params": best_xrfm_params,
            "test_metrics": xrfm_metrics,
            "agop_top_eigenvalue": agop_summary["top_eigenvalue"],
            "agop_index": agop_summary["best_agop_index"],
        },
        "xgboost": {
            "params": best_xgb_params,
            "test_metrics": xgb_metrics,
        },
    }

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    metrics_df, metrics_csv_path = make_metrics_csv(
        xrfm_metrics=xrfm_metrics,
        xgb_metrics=xgb_metrics,
        output_dir=output_dir,
    )

    print("\nTest metrics:")
    print(json.dumps(results, indent=2))
    print("\nSaved metrics to:", metrics_csv_path)
    print(metrics_df)

if __name__ == "__main__":
    main()
