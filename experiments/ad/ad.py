SEED = 42

import os
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json
import sys
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

    return {
        "top_diag_df": diag_df.head(top_k),
        "top_eigen_df": eigen_df.head(top_k),
        "top_eigenvalue": top_eigenvalue,
        "best_agop_index": best_index,
        "diag_path": diag_path,
        "eigen_path": eigen_path,
    }


def make_metrics_csv(xrfm_metrics, xgb_metrics, rf_metrics, output_dir):
    metrics_df = pd.DataFrame([
        {
            "model": "xrfm",
            "accuracy": xrfm_metrics["accuracy"],
            "roc_auc": xrfm_metrics.get("roc_auc", ""),
        },
        {
            "model": "xgboost",
            "accuracy": xgb_metrics["accuracy"],
            "roc_auc": xgb_metrics.get("roc_auc", ""),
        },
        {
            "model": "random_forest",
            "accuracy": rf_metrics["accuracy"],
            "roc_auc": rf_metrics.get("roc_auc", ""),
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

    print("Best xRFM params:", best_xrfm_params)
    print("Best XGB params:", best_xgb_params)
    print("Best RF params:", best_rf_params)

    X_train_np, X_val_np, X_test_np, y_train_np, y_val_np, y_test_np = to_numpy(
        X_train_df,
        X_val_df,
        X_test_df,
        y_train_s,
        y_val_s,
        y_test_s,
    )

    # Run xRFM first. Do not import XGBoost before this point.
    xrfm_model = xRFM(**best_xrfm_params, random_state=SEED)
    xrfm_model.fit(X_train_np, y_train_np, X_val=X_val_np, y_val=y_val_np)
    xrfm_metrics = evaluate_model(xrfm_model, X_test_np, y_test_np)

    agop_summary = extract_highest_agop_summary(
        model=xrfm_model,
        feature_names=list(X_train_df.columns),
        output_dir=output_dir,
        top_k=20,
    )

    # Import XGBoost only after xRFM has finished fitting.
    from xgboost import XGBClassifier

    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=1,
        **best_xgb_params,
    )

    xgb_model.fit(X_train_df, y_train_s)
    xgb_metrics = evaluate_model(xgb_model, X_test_df, y_test_s)

    # Import Random Forest only after XGBoost has finished.
    from sklearn.ensemble import RandomForestClassifier

    rf_model = RandomForestClassifier(
        random_state=SEED,
        n_jobs=1,
        **best_rf_params,
    )
    rf_model.fit(X_train_df, y_train_s)
    rf_metrics = evaluate_model(rf_model, X_test_df, y_test_s)

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
        "xrfm_agop": {
            "selected_agop_index": agop_summary["best_agop_index"],
            "selection_rule": "largest_top_eigenvalue",
            "top_eigenvalue": agop_summary["top_eigenvalue"],
            "diagonal_csv": str(agop_summary["diag_path"]),
            "top_eigenvector_loadings_csv": str(agop_summary["eigen_path"]),
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
    print("\nSaved metrics to:", metrics_csv_path)
    print(metrics_df)


if __name__ == "__main__":
    main()