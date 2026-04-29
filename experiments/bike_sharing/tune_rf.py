SEED = 42
THREADS = 1

import os
import sys
from pathlib import Path

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = str(THREADS)
os.environ["MKL_NUM_THREADS"] = str(THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(THREADS)

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.experiment import run_tuning_job
from experiments.bike_sharing.load_data import load_bike_splits
from src.tuning.rf_tuner_reg import tune_rf

def main():
    output_dir = ROOT / "outputs" / "bike_sharing"
    run_tuning_job(
        output_dir=output_dir,
        load_splits=load_bike_splits,
        tune_func=tune_rf,
        result_name="rf",
        seed=SEED,
    )


if __name__ == "__main__":
    main()
