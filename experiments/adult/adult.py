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
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xrfm import xRFM
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.preprocessing import preprocess_data
from experiments.adult.load_data import load_adult_splits

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

def clean_income_labels(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .map({"<=50K": 0, ">50K": 1})
    )

def load_best_params():
    output_dir = ROOT / "outputs" / "adult"

    with open(output_dir / "xrfm_best_params.json", "r") as f:
        xrfm_result = json.load(f)

    with open(output_dir / "xgb_best_params.json", "r") as f:
        xgb_result = json.load(f)

    return xrfm_result["params"], xgb_result["params"]

def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred))
    }

    if hasattr(model, "predict_proba"):
        y_score = np.asarray(model.predict_proba(X))

        if y_score.ndim == 2 and y_score.shape[1] >= 2:
            y_score = y_score[:, 1]
        elif y_score.ndim == 2 and y_score.shape[1] == 1:
            y_score = y_score[:, 0]

        metrics["roc_auc"] = float(roc_auc_score(y, y_score))

    return metrics

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

    # Save full AGOP matrix
    agop_df = pd.DataFrame(
        best_agop,
        index=feature_names,
        columns=feature_names,
    )

    agop_path = output_dir / "xrfm_best_agop.csv"
    agop_df.to_csv(agop_path)

    print("\nSaved full AGOP matrix to:", agop_path)

    if best_agop.shape[0] != len(feature_names):
        raise ValueError(
            f"AGOP dimension {best_agop.shape[0]} does not match "
            f"{len(feature_names)} feature names."
        )

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

    print("\nHighest AGOP index:", best_index)
    print("Highest AGOP top eigenvalue:", top_eigenvalue)

    print("\nTop AGOP diagonal features:")
    print(diag_df.head(top_k))

    print("\nTop AGOP eigenvector loadings:")
    print(eigen_df.head(top_k))

    return {
        "top_diag_df": diag_df.head(top_k),
        "top_eigen_df": eigen_df.head(top_k),
        "top_eigenvalue": top_eigenvalue,
        "best_agop_index": best_index,
        "diag_path": diag_path,
        "eigen_path": eigen_path,
    }

def make_metrics_csv(xrfm_metrics, xgb_metrics, agop_summary, output_dir):
    model_metrics_df = pd.DataFrame([
        {
            "row_type": "model_metric",
            "model": "xrfm",
            "accuracy": xrfm_metrics["accuracy"],
            "roc_auc": xrfm_metrics.get("roc_auc", ""),
            "agop_summary_type": "",
            "rank": "",
            "feature": "",
            "value": "",
            "abs_value": "",
            "top_eigenvalue": "",
            "selected_agop_index": "",
        },
        {
            "row_type": "model_metric",
            "model": "xgboost",
            "accuracy": xgb_metrics["accuracy"],
            "roc_auc": xgb_metrics.get("roc_auc", ""),
            "agop_summary_type": "",
            "rank": "",
            "feature": "",
            "value": "",
            "abs_value": "",
            "top_eigenvalue": "",
            "selected_agop_index": "",
        },
    ])

    top_diag_df = agop_summary["top_diag_df"]
    top_eigen_df = agop_summary["top_eigen_df"]
    top_eigenvalue = agop_summary["top_eigenvalue"]
    best_agop_index = agop_summary["best_agop_index"]

    agop_diag_df = pd.DataFrame([
        {
            "row_type": "agop_summary",
            "model": "xrfm",
            "accuracy": "",
            "roc_auc": "",
            "agop_summary_type": "top_diagonal",
            "rank": i + 1,
            "feature": row["feature"],
            "value": row["agop_diagonal"],
            "abs_value": row["abs_agop_diagonal"],
            "top_eigenvalue": top_eigenvalue,
            "selected_agop_index": best_agop_index,
        }
        for i, (_, row) in enumerate(top_diag_df.iterrows())
    ])

    agop_eigen_df = pd.DataFrame([
        {
            "row_type": "agop_summary",
            "model": "xrfm",
            "accuracy": "",
            "roc_auc": "",
            "agop_summary_type": "top_eigenvector_loading",
            "rank": i + 1,
            "feature": row["feature"],
            "value": row["top_eigenvector_loading"],
            "abs_value": row["abs_loading"],
            "top_eigenvalue": top_eigenvalue,
            "selected_agop_index": best_agop_index,
        }
        for i, (_, row) in enumerate(top_eigen_df.iterrows())
    ])

    metrics_df = model_metrics_df

    metrics_csv_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    return metrics_df, metrics_csv_path

def main():
    output_dir = ROOT / "outputs" / "adult"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s = load_adult_splits()

    print("Shapes:")
    print("X_train:", X_train_df.shape)
    print("X_val:", X_val_df.shape)
    print("X_test:", X_test_df.shape)

    best_xrfm_params, best_xgb_params = load_best_params()

    print("Best xRFM params:", best_xrfm_params)
    print("Best XGB params:", best_xgb_params)

    X_train_np = np.asarray(X_train_df, dtype=np.float32)
    X_val_np = np.asarray(X_val_df, dtype=np.float32)
    X_test_np = np.asarray(X_test_df, dtype=np.float32)

    y_train_np = np.asarray(y_train_s, dtype=np.int64)
    y_val_np = np.asarray(y_val_s, dtype=np.int64)
    y_test_np = np.asarray(y_test_s, dtype=np.int64)

    xrfm_model = xRFM(
        **best_xrfm_params,
        n_threads=N_THREADS,
        random_state=SEED,
    )

    xrfm_model.fit(
        X_train_np,
        y_train_np,
        X_val=X_val_np,
        y_val=y_val_np,
    )

    xrfm_metrics = evaluate_model(xrfm_model, X_test_np, y_test_np)

    agop_summary = extract_highest_agop_summary(
        model=xrfm_model,
        feature_names=list(X_train_df.columns),
        output_dir=output_dir,
        top_k=20,
    )

    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
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
        },
        "xgboost": {
            "params": best_xgb_params,
            "test_metrics": xgb_metrics,
        }
    }

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    metrics_df, metrics_csv_path = make_metrics_csv(
        xrfm_metrics=xrfm_metrics,
        xgb_metrics=xgb_metrics,
        agop_summary=agop_summary,
        output_dir=output_dir,
    )

    print("\nTest metrics:")
    print(json.dumps(results, indent=2))

    print("\nSaved combined metrics to:", metrics_csv_path)
    print(metrics_df)

if __name__ == "__main__":
    main()
