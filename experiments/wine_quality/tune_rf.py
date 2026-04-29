import os
import sys
from pathlib import Path
import numpy as np
from src.utils.experiment import run_tuning_job
from experiments.wine_quality.load_data import load_wine_splits
from src.tuning.rf_tuner_reg import tune_rf

SEED = 42
THREADS = 1

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
        tune_func=tune_rf,
        result_name="rf",
        seed=SEED,
    )

if __name__ == "__main__":
    main()
