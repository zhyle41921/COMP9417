import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from xrfm import xRFM

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from sklearn.model_selection import train_test_split

SEED = 42


def load_bike_sharing_data():
    data_path = ROOT / "experiments" / "bike_sharing" / "hour.csv"

    df = pd.read_csv(
        data_path,
        header=0,
        na_values="?",
        skipinitialspace=True,
        low_memory=False,
    )

    # Drop non-informative columns
    # 'instant' is just a row index
    # 'dteday' is a raw date string
    # 'casual' and 'registered' sum directly to 'cnt' — data leakage
    df = df.drop(columns=["instant", "dteday", "casual", "registered"])

    return df


# Default hyperparameters used when no tuned params file exists yet
DEFAULT_XRFM_PARAMS = {"n_estimators": 100, "max_depth": 5}
DEFAULT_XGB_PARAMS = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}


def load_best_params():
    output_dir = ROOT / "outputs" / "bike_sharing"
    output_dir.mkdir(parents=True, exist_ok=True)

    xrfm_path = output_dir / "xrfm_best_params.json"
    xgb_path = output_dir / "xgb_best_params.json"

    # If the file exists, load it; otherwise create it with default params
    if xrfm_path.exists():
        with open(xrfm_path, "r") as f:
            xrfm_result = json.load(f)
        xrfm_params = xrfm_result["params"]
    else:
        print(f"xRFM params not found — creating {xrfm_path} with defaults")
        xrfm_params = DEFAULT_XRFM_PARAMS
        with open(xrfm_path, "w") as f:
            json.dump({"params": xrfm_params}, f, indent=2)

    if xgb_path.exists():
        with open(xgb_path, "r") as f:
            xgb_result = json.load(f)
        xgb_params = xgb_result["params"]
    else:
        print(f"XGB params not found — creating {xgb_path} with defaults")
        xgb_params = DEFAULT_XGB_PARAMS
        with open(xgb_path, "w") as f:
            json.dump({"params": xgb_params}, f, indent=2)

    return xrfm_params, xgb_params


def to_numpy(X_train, X_val, X_test, y_train, y_val, y_test):
    return (
        np.asarray(X_train, dtype=np.float32),
        np.asarray(X_val, dtype=np.float32),
        np.asarray(X_test, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),  # float for regression
        np.asarray(y_val, dtype=np.float32),
        np.asarray(y_test, dtype=np.float32),
    )


def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    r2 = float(r2_score(y, y_pred))

    metrics = {
        "rmse": rmse,
        "r2": r2,
    }

    return metrics


def main():
    df = load_bike_sharing_data()

    X = df.drop(columns=["cnt"])
    y = df["cnt"]

    # First split off the test set (20%), then split the remainder into train/val (80/20)
    X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    X_train_df, X_val_df, y_train_s, y_val_s = train_test_split(
        X_train_df, y_train_s, test_size=0.2, random_state=SEED
    )

    best_xrfm_params, best_xgb_params = load_best_params()

    print("Best xRFM params:", best_xrfm_params)
    print("Best XGB params:", best_xgb_params)

    X_train_np, X_val_np, X_test_np, y_train_np, y_val_np, y_test_np = to_numpy(
        X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s
    )

    xrfm_model = xRFM(**best_xrfm_params, random_state=SEED)
    xrfm_model.fit(X_train_np, y_train_np, X_val=X_val_np, y_val=y_val_np)
    xrfm_metrics = evaluate_model(xrfm_model, X_test_np, y_test_np)

    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=SEED,
        n_jobs=1,
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
        },
    }

    output_dir = ROOT / "outputs" / "bike_sharing"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "test_metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    metrics_df = pd.DataFrame([
        {
            "model": "xrfm",
            "rmse": results["xrfm"]["test_metrics"]["rmse"],
            "r2": results["xrfm"]["test_metrics"]["r2"],
        },
        {
            "model": "xgboost",
            "rmse": results["xgboost"]["test_metrics"]["rmse"],
            "r2": results["xgboost"]["test_metrics"]["r2"],
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