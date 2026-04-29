import os
import sys
from pathlib import Path
import numpy as np
from src.utils.experiment import run_tuning_job
from experiments.wine_quality.load_data import load_wine_splits
from src.tuning.xrfm_tuner_reg import tune_xrfm_regression

SEED = 42
THREADS = 4

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = str(THREADS)
os.environ["MKL_NUM_THREADS"] = str(THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(THREADS)

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

def main():
    output_dir = ROOT / "outputs" / "wine_quality"
    run_tuning_job(
        output_dir=output_dir,
        load_splits=load_wine_splits,
        tune_func=tune_xrfm_regression,
        result_name="xrfm",
        seed=SEED,
        x_dtype=np.float32,
        y_dtype=np.float32,
    )

if __name__ == "__main__":
    main()
