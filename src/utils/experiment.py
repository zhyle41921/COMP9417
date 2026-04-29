import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def load_best_params(output_dir):
    output_dir = Path(output_dir)
    names = ("xrfm", "xgb", "rf")
    params = []

    for name in names:
        with open(output_dir / f"{name}_best_params.json", "r") as f:
            params.append(json.load(f)["params"])

    return tuple(params)


def fit_with_time(model, *fit_args, **fit_kwargs):
    start = time.perf_counter()
    model.fit(*fit_args, **fit_kwargs)
    return model, float(time.perf_counter() - start)


def to_numpy_splits(splits, x_dtype=np.float32, y_dtype=np.float32):
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    return (
        np.asarray(X_train, dtype=x_dtype),
        np.asarray(X_val, dtype=x_dtype),
        np.asarray(X_test, dtype=x_dtype),
        np.asarray(y_train, dtype=y_dtype),
        np.asarray(y_val, dtype=y_dtype),
        np.asarray(y_test, dtype=y_dtype),
    )


def evaluate_classification(model, X, y, include_total_time=False):
    start = time.perf_counter()
    y_pred = model.predict(X)
    inference_time = time.perf_counter() - start

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "inference_time_per_sample_seconds": float(inference_time / len(y)),
    }

    if include_total_time:
        metrics["inference_time_seconds"] = float(inference_time)

    if hasattr(model, "predict_proba"):
        y_score = np.asarray(model.predict_proba(X))
        if y_score.ndim == 2 and y_score.shape[1] >= 2:
            y_score = y_score[:, 1]
        elif y_score.ndim == 2 and y_score.shape[1] == 1:
            y_score = y_score[:, 0]

        metrics["roc_auc"] = float(roc_auc_score(y, y_score))

    return metrics


def evaluate_regression(model, X, y, include_full_metrics=False, include_total_time=False):
    start = time.perf_counter()
    y_pred = model.predict(X)
    inference_time = time.perf_counter() - start
    mse = mean_squared_error(y, y_pred)

    metrics = {
        "rmse": float(np.sqrt(mse)),
    }

    if include_full_metrics:
        metrics.update({
            "mse": float(mse),
            "mae": float(mean_absolute_error(y, y_pred)),
            "r2": float(r2_score(y, y_pred)),
        })

    if include_total_time:
        metrics["inference_time_seconds"] = float(inference_time)
    else:
        metrics["inference_time_per_sample_seconds"] = float(inference_time / len(y))

    return metrics


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_metrics_csv(rows, output_dir):
    metrics_df = pd.DataFrame(rows)
    metrics_csv_path = Path(output_dir) / "metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    return metrics_df, metrics_csv_path


def metric_row(model_name, metrics, fields, extra=None):
    row = {"model": model_name}
    if extra:
        row.update(extra)
    row.update({field: metrics.get(field, "") for field in fields})
    return row


def print_shapes(X_train, X_val, X_test):
    print("Shapes:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)


def run_tuning_job(
    output_dir,
    load_splits,
    tune_func,
    result_name,
    seed,
    x_dtype=None,
    y_dtype=None,
    param_grid=None,
    print_columns=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    if print_columns:
        print("Columns after preprocessing:")
        print(list(X_train.columns))

    if x_dtype is not None or y_dtype is not None:
        X_train, X_val, X_test, y_train, y_val, y_test = to_numpy_splits(
            (X_train, X_val, X_test, y_train, y_val, y_test),
            x_dtype=x_dtype or np.float32,
            y_dtype=y_dtype or np.float32,
        )

    print_shapes(X_train, X_val, X_test)

    kwargs = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "results_path": output_dir / f"{result_name}_results.json",
        "best_path": output_dir / f"{result_name}_best_params.json",
        "seed": seed,
    }

    if param_grid is not None:
        kwargs["param_grid"] = param_grid

    best_result, results = tune_func(**kwargs)

    print("\nBest result:")
    print(json.dumps(best_result, indent=2))

    return best_result, results
