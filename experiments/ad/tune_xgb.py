SEED = 42

import os
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import random
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(SEED)
np.random.seed(SEED)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from src.tuning.xgb_tuner import tune_xgb
from experiments.ad.load_data import load_ad_splits

def main():
    ROOT = Path(__file__).resolve().parents[2]

    output_dir = ROOT / "outputs" / "ad"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "xgb_results.json"
    best_path = output_dir / "xgb_best_params.json"

    X_train, X_val, X_test, y_train, y_val, y_test = load_ad_splits()

    print("Shapes:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    best_result, results = tune_xgb(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        results_path=results_path,
        best_path=best_path,
        seed=SEED,
    )

    print("\nBest result:")
    print(json.dumps(best_result, indent=2))

if __name__ == "__main__":
    main()
