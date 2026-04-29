SEED = 42

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.preprocessing import preprocess_data

def load_bike_sharing_data():
    data_path = ROOT / "experiments" / "bike_sharing" / "data" / "hour.csv"

    df = pd.read_csv(
        data_path,
        header=0,
        na_values="?",
        skipinitialspace=True,
        low_memory=False,
    )

    # Keep it simple: drop leakage and non-useful ID/date columns.
    df = df.drop(columns=["instant", "dteday", "casual", "registered"])

    return df

def save_bike_splits(seed=SEED):
    output_dir = ROOT / "experiments" / "bike_sharing" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_bike_sharing_data()

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        df,
        target_col="cnt",
        random_state=seed,
        stratify=False,
        do_remove_duplicates=False,
    )

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_val.to_csv(output_dir / "X_val.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)

    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_val.to_csv(output_dir / "y_val.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)


def load_bike_splits():
    split_dir = ROOT / "experiments" / "bike_sharing" / "data"
    required_files = [
        split_dir / "X_train.csv",
        split_dir / "X_val.csv",
        split_dir / "X_test.csv",
        split_dir / "y_train.csv",
        split_dir / "y_val.csv",
        split_dir / "y_test.csv",
    ]

    if not all(path.exists() for path in required_files):
        save_bike_splits(seed=SEED)

    X_train = pd.read_csv(split_dir / "X_train.csv")
    X_val = pd.read_csv(split_dir / "X_val.csv")
    X_test = pd.read_csv(split_dir / "X_test.csv")

    y_train = pd.read_csv(split_dir / "y_train.csv").squeeze("columns")
    y_val = pd.read_csv(split_dir / "y_val.csv").squeeze("columns")
    y_test = pd.read_csv(split_dir / "y_test.csv").squeeze("columns")

    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    save_bike_splits(seed=SEED)

if __name__ == "__main__":
    main()
