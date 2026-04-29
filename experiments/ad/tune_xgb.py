import os
import sys
from pathlib import Path
import numpy as np
from src.utils.experiment import run_tuning_job
from experiments.ad.load_data import load_ad_splits
from src.tuning.xgb_tuner import tune_xgb

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
    output_dir = ROOT / "outputs" / "ad"
    run_tuning_job(
        output_dir=output_dir,
        load_splits=load_ad_splits,
        tune_func=tune_xgb,
        result_name="xgb",
        seed=SEED,
    )

if __name__ == "__main__":
    main()
