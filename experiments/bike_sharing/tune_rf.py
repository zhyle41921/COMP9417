SEED = 42

import os
os.environ["PYTHONHASHSEED"] = str(SEED)

import random
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.bike_sharing.load_data import load_bike_splits
from src.tuning.rf_tuner_reg import tune_rf

random.seed(SEED)
np.random.seed(SEED)

def main():
    output_dir = ROOT / "outputs" / "bike_sharing"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "rf_results.json"
    best_path = output_dir / "rf_best_params.json"

    X_train, X_val, X_test, y_train, y_val, y_test = load_bike_splits()

    print("Shapes:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    best_result, results = tune_rf(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        results_path=results_path,
        best_path=best_path,
        task="regression",
        seed=SEED,
    )

    print("\nBest RF result:")
    print(json.dumps(best_result, indent=2))

if __name__ == "__main__":
    main()
