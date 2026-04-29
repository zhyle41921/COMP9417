SEED = 42

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.preprocessing import preprocess_data

def load_ad_data():
    n_features = 1558
    col_names = [f"x{i}" for i in range(n_features)] + ["label"]

    data_path = ROOT / "experiments" / "ad" / "data" / "ad.data"

    df = pd.read_csv(
        data_path,
        header=None,
        names=col_names,
        na_values="?",
        skipinitialspace=True,
        low_memory=False,
    )

    df["label"] = df["label"].str.strip()
    df["label"] = df["label"].map({"nonad.": 0, "ad.": 1})

    if df["label"].isna().any():
        raise ValueError("Found unmapped labels in ad dataset.")

    return df

def save_ad_splits(seed=SEED):
    output_dir = ROOT / "experiments" / "ad" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_ad_data()

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        df,
        target_col="label",
        random_state=seed,
        do_scale=False,
        do_impute=False,
        do_dropna=True
    )

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_val.to_csv(output_dir / "X_val.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)

    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_val.to_csv(output_dir / "y_val.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)


def load_ad_splits():
    split_dir = ROOT / "experiments" / "ad" / "data"

    X_train = pd.read_csv(split_dir / "X_train.csv")
    X_val = pd.read_csv(split_dir / "X_val.csv")
    X_test = pd.read_csv(split_dir / "X_test.csv")

    y_train = pd.read_csv(split_dir / "y_train.csv").squeeze("columns")
    y_val = pd.read_csv(split_dir / "y_val.csv").squeeze("columns")
    y_test = pd.read_csv(split_dir / "y_test.csv").squeeze("columns")

    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    save_ad_splits(seed=SEED)

if __name__ == "__main__":
    main()
