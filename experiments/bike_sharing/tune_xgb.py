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

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.tuning.xgb_tuner_reg import tune_xgb_regression
from experiments.bike_sharing.load_data import load_bike_splits


random.seed(SEED)
np.random.seed(SEED)


def main():
    output_dir = ROOT / "outputs" / "bike_sharing"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_bike_splits()

    print("Column names after preprocessing:")
    print(list(X_train.columns))

    print("Shapes:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    best_result, results = tune_xgb_regression(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        results_path=output_dir / "xgb_results.json",
        best_path=output_dir / "xgb_best_params.json",
        seed=SEED,
    )

    print("\nBest XGB regression result:")
    print(json.dumps(best_result, indent=2))


if __name__ == "__main__":
    main()
