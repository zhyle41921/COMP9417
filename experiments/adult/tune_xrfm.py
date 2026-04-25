SEED = 42

import os

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

import random
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.preprocessing import preprocess_data
from src.tuning.xrfm_tuner import tune_xrfm


random.seed(SEED)
np.random.seed(SEED)


COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]


def load_adult_data():
    data_path = ROOT / "experiments" / "adult" / "adult.data"

    df = pd.read_csv(
        data_path,
        header=None,
        names=COLUMNS,
        na_values="?",
        skipinitialspace=True,
        low_memory=False,
    )

    df["income"] = df["income"].str.strip()
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})

    if df["income"].isna().any():
        raise ValueError("Found unmapped labels in adult dataset.")

    return df


def to_numpy(X_train, X_val, X_test, y_train, y_val, y_test):
    return (
        np.asarray(X_train, dtype=np.float32),
        np.asarray(X_val, dtype=np.float32),
        np.asarray(X_test, dtype=np.float32),
        np.asarray(y_train, dtype=np.int64),
        np.asarray(y_val, dtype=np.int64),
        np.asarray(y_test, dtype=np.int64),
    )


def main():
    output_dir = ROOT / "outputs" / "adult"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_adult_data()

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        df,
        target_col="income",
        random_state=SEED,
        do_remove_duplicates=False,
    )

    print("Columns after preprocessing:")
    print(list(X_train.columns))

    X_train, X_val, X_test, y_train, y_val, y_test = to_numpy(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

    print("Shapes:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    best_result, results = tune_xrfm(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        results_path=output_dir / "xrfm_results.json",
        best_path=output_dir / "xrfm_best_params.json",
        seed=SEED,
        max_leaf_size_values=[1024, 2048, 4096]
    )

    print("\nBest result:")
    print(json.dumps(best_result, indent=2))


if __name__ == "__main__":
    main()