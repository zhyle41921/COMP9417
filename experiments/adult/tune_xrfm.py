SEED = 42
THREADS = 4

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
from experiments.adult.load_data import load_adult_splits
from src.tuning.xrfm_tuner import tune_xrfm


def main():
    output_dir = ROOT / "outputs" / "adult"
    run_tuning_job(
        output_dir=output_dir,
        load_splits=load_adult_splits,
        tune_func=tune_xrfm,
        result_name="xrfm",
        seed=SEED,
        x_dtype=np.float32,
        y_dtype=np.int64,
        param_grid={"max_leaf_size": [2048, 4096, 8192]},
    )


if __name__ == "__main__":
    main()
