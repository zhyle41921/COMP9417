# experiments/adult/load_data.py

SEED = 42

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

NUMERIC_COLS = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

def clean_income_labels(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .map({"<=50K": 0, ">50K": 1})
    )

def group_adult_categories(df):
    df = df.copy()

    df = df.fillna("Unknown")

    df["education_group"] = df["education"].replace({
        "Preschool": "school_before_hs",
        "1st-4th": "school_before_hs",
        "5th-6th": "school_before_hs",
        "7th-8th": "school_before_hs",
        "9th": "school_before_hs",
        "10th": "school_before_hs",
        "11th": "school_before_hs",
        "12th": "school_before_hs",
        "HS-grad": "high_school",
        "Some-college": "some_college",
        "Assoc-acdm": "associate",
        "Assoc-voc": "associate",
        "Bachelors": "bachelors",
        "Masters": "masters",
        "Doctorate": "doctorate",
        "Prof-school": "professional_school",
        "Unknown": "unknown",
    })

    df["workclass_group"] = df["workclass"].replace({
        "Private": "private",
        "Self-emp-not-inc": "self_employed",
        "Self-emp-inc": "self_employed",
        "Federal-gov": "government",
        "Local-gov": "government",
        "State-gov": "government",
        "Without-pay": "unpaid_or_never_worked",
        "Never-worked": "unpaid_or_never_worked",
        "Unknown": "unknown",
    })

    df["marital_group"] = df["marital_status"].replace({
        "Married-civ-spouse": "married",
        "Married-AF-spouse": "married",
        "Married-spouse-absent": "married_absent",
        "Divorced": "previously_married",
        "Separated": "previously_married",
        "Widowed": "previously_married",
        "Never-married": "never_married",
        "Unknown": "unknown",
    })

    df["occupation_group"] = df["occupation"].replace({
        "Exec-managerial": "managerial_professional",
        "Prof-specialty": "managerial_professional",
        "Tech-support": "technical_support",
        "Sales": "sales_admin",
        "Adm-clerical": "sales_admin",
        "Craft-repair": "blue_collar",
        "Machine-op-inspct": "blue_collar",
        "Transport-moving": "blue_collar",
        "Handlers-cleaners": "blue_collar",
        "Farming-fishing": "blue_collar",
        "Protective-serv": "service",
        "Other-service": "service",
        "Priv-house-serv": "service",
        "Armed-Forces": "armed_forces",
        "Unknown": "unknown",
    })

    df["relationship_group"] = df["relationship"].replace({
        "Husband": "spouse",
        "Wife": "spouse",
        "Own-child": "child",
        "Other-relative": "other_relative",
        "Not-in-family": "not_in_family",
        "Unmarried": "unmarried",
        "Unknown": "unknown",
    })

    df["native_country_group"] = df["native_country"].replace({
        "United-States": "united_states",

        "Canada": "north_america",
        "Mexico": "north_america",
        "Outlying-US(Guam-USVI-etc)": "north_america",

        "Puerto-Rico": "central_america_caribbean",
        "Cuba": "central_america_caribbean",
        "Jamaica": "central_america_caribbean",
        "Dominican-Republic": "central_america_caribbean",
        "Haiti": "central_america_caribbean",
        "Honduras": "central_america_caribbean",
        "Guatemala": "central_america_caribbean",
        "Nicaragua": "central_america_caribbean",
        "El-Salvador": "central_america_caribbean",
        "Trinadad&Tobago": "central_america_caribbean",

        "Columbia": "south_america",
        "Ecuador": "south_america",
        "Peru": "south_america",

        "England": "europe",
        "Germany": "europe",
        "Greece": "europe",
        "Italy": "europe",
        "Poland": "europe",
        "Portugal": "europe",
        "Ireland": "europe",
        "France": "europe",
        "Hungary": "europe",
        "Scotland": "europe",
        "Yugoslavia": "europe",
        "Holand-Netherlands": "europe",

        "India": "asia",
        "Japan": "asia",
        "China": "asia",
        "Iran": "asia",
        "Philippines": "asia",
        "Vietnam": "asia",
        "Laos": "asia",
        "Taiwan": "asia",
        "Thailand": "asia",
        "Cambodia": "asia",
        "Hong": "asia",

        "South": "other",
        "Unknown": "unknown",
    })

    df = df.drop(columns=[
        "education",
        "workclass",
        "marital_status",
        "occupation",
        "relationship",
        "native_country",
    ])

    return df

def load_adult_file(filename):
    data_path = ROOT / "experiments" / "adult" / "data" / filename

    print("Reading:", data_path)

    df = pd.read_csv(
        data_path,
        header=None,
        names=COLUMNS,
        na_values="?",
        skipinitialspace=True,
        low_memory=False,
        comment="|",
    )

    df["income"] = clean_income_labels(df["income"])

    if df["income"].isna().any():
        raise ValueError(f"Found unmapped labels in {filename}.")

    df = df.drop(columns=["fnlwgt"])
    df = group_adult_categories(df)

    return df

def preprocess_adult_combined(combined_df):
    combined_df = combined_df.copy()

    y_all = combined_df["income"].astype(int)
    X_all = combined_df.drop(columns=["income"])

    cat_cols = [col for col in X_all.columns if col not in NUMERIC_COLS]

    X_all = pd.get_dummies(
        X_all,
        columns=cat_cols,
        drop_first=False,
        dtype=int,
    )

    scaler = StandardScaler()
    X_all[NUMERIC_COLS] = scaler.fit_transform(X_all[NUMERIC_COLS])

    return X_all, y_all

def save_adult_splits(seed=SEED):
    output_dir = ROOT / "experiments" / "adult" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_adult_file("adult.data")
    test_df = load_adult_file("adult.test")

    n_train = len(train_df)

    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    X_all, y_all = preprocess_adult_combined(combined_df)

    X_train_full = X_all.iloc[:n_train]
    y_train_full = y_all.iloc[:n_train]

    X_test = X_all.iloc[n_train:]
    y_test = y_all.iloc[n_train:]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=seed,
        stratify=y_train_full,
    )

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_val.to_csv(output_dir / "X_val.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)

    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_val.to_csv(output_dir / "y_val.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print("Saved fixed Adult splits to:", output_dir)
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    print("One-hot check:")
    print(X_train.filter(like="workclass_group_").sum(axis=1).value_counts().head())

    for prefix in [
        "race_",
        "sex_",
        "education_group_",
        "workclass_group_",
        "marital_group_",
        "occupation_group_",
        "relationship_group_",
        "native_country_group_",
    ]:
        cols = [c for c in X_train.columns if c.startswith(prefix)]
        print(prefix, X_train[cols].sum(axis=1).value_counts().head())

def load_adult_splits():
    split_dir = ROOT / "experiments" / "adult" / "data"

    X_train = pd.read_csv(split_dir / "X_train.csv")
    X_val = pd.read_csv(split_dir / "X_val.csv")
    X_test = pd.read_csv(split_dir / "X_test.csv")

    y_train = pd.read_csv(split_dir / "y_train.csv").squeeze("columns")
    y_val = pd.read_csv(split_dir / "y_val.csv").squeeze("columns")
    y_test = pd.read_csv(split_dir / "y_test.csv").squeeze("columns")

    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    save_adult_splits(seed=SEED)

if __name__ == "__main__":
    main()
