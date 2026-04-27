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

from src.tuning.xrfm_tuner_reg import tune_xrfm_regression
from experiments.wine_quality.load_data import load_wine_splits

random.seed(SEED)
np.random.seed(SEED)

def to_numpy(X_train, X_val, X_test, y_train, y_val, y_test):
    return (
        np.asarray(X_train, dtype=np.float32),
        np.asarray(X_val, dtype=np.float32),
        np.asarray(X_test, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
        np.asarray(y_val, dtype=np.float32),
        np.asarray(y_test, dtype=np.float32),
    )

def main():
    output_dir = ROOT / "outputs" / "wine_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_wine_splits()

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

    best_result, results = tune_xrfm_regression(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        results_path=output_dir / "xrfm_results.json",
        best_path=output_dir / "xrfm_best_params.json",
        seed=SEED
    )

    print("\nBest result:")
    print(json.dumps(best_result, indent=2))

if __name__ == "__main__":
    main()
