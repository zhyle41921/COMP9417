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

from src.utils.preprocessing import preprocess_data


def load_ad_data():
    n_features = 1558
    col_names = [f"x{i}" for i in range(n_features)] + ["label"]

    data_path = ROOT / "experiments" / "ad" / "ad.data"

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


def main():
    output_dir = ROOT / "outputs" / "ad"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_ad_data()

    X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s = preprocess_data(
        df,
        target_col="label",
        random_state=SEED,
    )

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
    }

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    metrics_df = pd.DataFrame([
        {
            "model": "xrfm",
            "accuracy": xrfm_metrics["accuracy"],
            "roc_auc": xrfm_metrics["roc_auc"],
        },
        {
            "model": "xgboost",
            "accuracy": xgb_metrics["accuracy"],
            "roc_auc": xgb_metrics["roc_auc"],
        },
        {
            "model": "random_forest",
            "accuracy": rf_metrics["accuracy"],
            "roc_auc": rf_metrics["roc_auc"],
        },
    ])

    metrics_csv_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    print("\nTest metrics:")
    print(json.dumps(results, indent=2))
    print("\nSaved metrics to:", metrics_csv_path)
    print(metrics_df)


if __name__ == "__main__":
    main()