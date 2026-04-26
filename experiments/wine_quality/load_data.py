SEED = 42

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.preprocessing import preprocess_data


def load_wine_data():
    data_dir = ROOT / "experiments" / "wine_quality" / "data"

    red_df = pd.read_csv(data_dir / "winequality-red.csv", sep=";")
    white_df = pd.read_csv(data_dir / "winequality-white.csv", sep=";")

    red_df["type"] = "red"
    white_df["type"] = "white"

    df = pd.concat([red_df, white_df], axis=0, ignore_index=True)

    df["quality"] = pd.to_numeric(df["quality"], errors="raise")

    return df


def save_wine_splits(seed=SEED):
    output_dir = ROOT / "experiments" / "wine_quality" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_wine_data()

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        df,
        target_col="quality",
        random_state=seed,
        do_remove_duplicates=False,
    )

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_val.to_csv(output_dir / "X_val.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)

    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_val.to_csv(output_dir / "y_val.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print("Saved fixed wine_quality splits to:", output_dir)
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)


def load_wine_splits():
    split_dir = ROOT / "experiments" / "wine_quality" / "data"

    X_train = pd.read_csv(split_dir / "X_train.csv")
    X_val = pd.read_csv(split_dir / "X_val.csv")
    X_test = pd.read_csv(split_dir / "X_test.csv")

    y_train = pd.read_csv(split_dir / "y_train.csv").squeeze("columns")
    y_val = pd.read_csv(split_dir / "y_val.csv").squeeze("columns")
    y_test = pd.read_csv(split_dir / "y_test.csv").squeeze("columns")

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    save_wine_splits(seed=SEED)


if __name__ == "__main__":
    main()