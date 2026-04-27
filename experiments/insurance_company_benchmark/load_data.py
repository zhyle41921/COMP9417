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
        / "data"
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

    rename_map = {
        "MOSTYPE": "customer_subtype",
        "MAANTHUI": "number_of_houses",
        "MGEMOMV": "avg_household_size",
        "MGEMLEEF": "avg_age_group",
        "MOSHOOFD": "customer_main_type",

        "MGODRK": "pct_roman_catholic",
        "MGODPR": "pct_protestant",
        "MGODOV": "pct_other_religion",
        "MGODGE": "pct_no_religion",

        "MRELGE": "pct_married",
        "MRELSA": "pct_living_together",
        "MRELOV": "pct_other_relationship",
        "MFALLEEN": "pct_singles",
        "MFGEKIND": "pct_households_without_children",
        "MFWEKIND": "pct_households_with_children",

        "MOPLHOOG": "pct_high_education",
        "MOPLMIDD": "pct_medium_education",
        "MOPLLAAG": "pct_low_education",

        "MBERHOOG": "pct_high_status",
        "MBERZELF": "pct_entrepreneurs",
        "MBERBOER": "pct_farmers",
        "MBERMIDD": "pct_middle_management",
        "MBERARBG": "pct_skilled_labourers",
        "MBERARBO": "pct_unskilled_labourers",

        "MSKA": "pct_social_class_a",
        "MSKB1": "pct_social_class_b1",
        "MSKB2": "pct_social_class_b2",
        "MSKC": "pct_social_class_c",
        "MSKD": "pct_social_class_d",

        "MHHUUR": "pct_rented_house",
        "MHKOOP": "pct_home_owners",

        "MAUT1": "pct_one_car",
        "MAUT2": "pct_two_cars",
        "MAUT0": "pct_no_car",

        "MZFONDS": "pct_national_health_service",
        "MZPART": "pct_private_health_insurance",

        "MINKM30": "pct_income_under_30k",
        "MINK3045": "pct_income_30k_to_45k",
        "MINK4575": "pct_income_45k_to_75k",
        "MINK7512": "pct_income_75k_to_122k",
        "MINK123M": "pct_income_over_123k",
        "MINKGEM": "avg_income_group",
        "MKOOPKLA": "purchasing_power_class",

        "PWAPART": "contribution_private_third_party_insurance",
        "PWABEDR": "contribution_firm_third_party_insurance",
        "PWALAND": "contribution_agriculture_third_party_insurance",
        "PPERSAUT": "contribution_car_policies",
        "PBESAUT": "contribution_delivery_van_policies",
        "PMOTSCO": "contribution_motorcycle_scooter_policies",
        "PVRAAUT": "contribution_lorry_policies",
        "PAANHANG": "contribution_trailer_policies",
        "PTRACTOR": "contribution_tractor_policies",
        "PWERKT": "contribution_agricultural_machine_policies",
        "PBROM": "contribution_moped_policies",
        "PLEVEN": "contribution_life_insurance",
        "PPERSONG": "contribution_private_accident_insurance",
        "PGEZONG": "contribution_family_accident_insurance",
        "PWAOREG": "contribution_disability_insurance",
        "PBRAND": "contribution_fire_policies",
        "PZEILPL": "contribution_surfboard_policies",
        "PPLEZIER": "contribution_boat_policies",
        "PFIETS": "contribution_bicycle_policies",
        "PINBOED": "contribution_property_insurance",
        "PBYSTAND": "contribution_social_security_insurance",

        "AWAPART": "num_private_third_party_insurance",
        "AWABEDR": "num_firm_third_party_insurance",
        "AWALAND": "num_agriculture_third_party_insurance",
        "APERSAUT": "num_car_policies",
        "ABESAUT": "num_delivery_van_policies",
        "AMOTSCO": "num_motorcycle_scooter_policies",
        "AVRAAUT": "num_lorry_policies",
        "AAANHANG": "num_trailer_policies",
        "ATRACTOR": "num_tractor_policies",
        "AWERKT": "num_agricultural_machine_policies",
        "ABROM": "num_moped_policies",
        "ALEVEN": "num_life_insurance",
        "APERSONG": "num_private_accident_insurance",
        "AGEZONG": "num_family_accident_insurance",
        "AWAOREG": "num_disability_insurance",
        "ABRAND": "num_fire_policies",
        "AZEILPL": "num_surfboard_policies",
        "APLEZIER": "num_boat_policies",
        "AFIETS": "num_bicycle_policies",
        "AINBOED": "num_property_insurance",
        "ABYSTAND": "num_social_security_insurance",

        "CARAVAN": "mobile_home_policy",
    }

    df = df.rename(columns=rename_map)

    df["mobile_home_policy"] = (
        pd.to_numeric(df["mobile_home_policy"], errors="raise")
        .astype(int)
    )

    return df

def save_insurance_splits(seed=SEED):
    output_dir = ROOT / "experiments" / "insurance_company_benchmark" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_insurance_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        df,
        target_col="mobile_home_policy",
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
