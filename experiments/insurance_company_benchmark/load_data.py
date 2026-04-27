SEED = 42

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.preprocessing import preprocess_data

def load_insurance_data():
    data_path = (
        ROOT
        / "experiments"
        / "insurance_company_benchmark"
        / "TICDATA_TICEVAL_combined.csv"
    )

    df = pd.read_csv(
        data_path,
        header=0,
        na_values="?",
        skipinitialspace=True,
        low_memory=False,
    )

    # These are metadata columns, not model features.
    drop_cols = [c for c in ["SOURCE_FILE", "SOURCE_ROW"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df["CARAVAN"] = pd.to_numeric(df["CARAVAN"], errors="raise").astype(int)
    return df

def save_insurance_splits(seed=SEED):
    output_dir = ROOT / "experiments" / "insurance_company_benchmark" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_insurance_data()

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        df,
        target_col="CARAVAN",
        random_state=seed,
        do_remove_duplicates=False,
        do_impute=False,
        do_scale=False,
        do_encode=False,
        stratify=True,
    )

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_val.to_csv(output_dir / "X_val.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)

    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_val.to_csv(output_dir / "y_val.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print("Saved fixed insurance splits to:", output_dir)
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

def load_insurance_splits():
    split_dir = ROOT / "experiments" / "insurance_company_benchmark" / "data"
    required_files = [
        split_dir / "X_train.csv",
        split_dir / "X_val.csv",
        split_dir / "X_test.csv",
        split_dir / "y_train.csv",
        split_dir / "y_val.csv",
        split_dir / "y_test.csv",
    ]

    if not all(path.exists() for path in required_files):
        print("Split files not found. Creating them now...")
        save_insurance_splits(seed=SEED)

    X_train = pd.read_csv(split_dir / "X_train.csv")
    X_val = pd.read_csv(split_dir / "X_val.csv")
    X_test = pd.read_csv(split_dir / "X_test.csv")

    y_train = pd.read_csv(split_dir / "y_train.csv").squeeze("columns")
    y_val = pd.read_csv(split_dir / "y_val.csv").squeeze("columns")
    y_test = pd.read_csv(split_dir / "y_test.csv").squeeze("columns")

    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    save_insurance_splits(seed=SEED)

if __name__ == "__main__":
    main()
