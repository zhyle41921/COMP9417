import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

def remove_duplicates(df):
    return df.drop_duplicates().copy()

def split_features_target(df, target_col):
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y

def split_data(X, y, val_size=0.2, test_size=0.2, random_state=RANDOM_STATE, stratify=True):
    stratify_labels = None
    if stratify:
        stratify_labels = y

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=val_size + test_size,
        stratify=stratify_labels,
        random_state=random_state,
    )

    test_ratio = test_size / (val_size + test_size)
    stratify_temp = None
    if stratify:
        stratify_temp = y_temp

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_ratio,
        stratify=stratify_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def drop_na_before_split(X, y):
    mask = X.notna().all(axis=1)
    return X.loc[mask].copy(), y.loc[mask].copy()

def impute_missing_values(X):
    X = X.copy()

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns

    for col in categorical_cols:
        fill_value = X[col].mode(dropna=True)[0]
        X[col] = X[col].fillna(fill_value)

    for col in numeric_cols:
        fill_value = X[col].mean()
        X[col] = X[col].fillna(fill_value)

    return X

def scale_numeric_features(X):
    X = X.copy()

    numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns

    if len(numeric_cols) == 0:
        return X

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X

def encode_categorical(X, drop_first=False):
    X = X.copy()

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    if len(categorical_cols) == 0:
        return X

    X = pd.get_dummies(
        X,
        columns=categorical_cols,
        drop_first=drop_first,
        dtype=int,
    )

    return X

def preprocess_data(
    df,
    target_col,
    val_size=0.2,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=True,
    do_remove_duplicates=True,
    do_impute=True,
    do_encode=True,
    do_scale=True,
    do_split=True,
    do_dropna=False,
    drop_first=False,
):
    df = df.copy()

    if do_remove_duplicates:
        df = remove_duplicates(df)

    X, y = split_features_target(df, target_col)

    if do_dropna:
        X, y = drop_na_before_split(X, y)

    if do_impute:
        X = impute_missing_values(X)

    # Scale BEFORE encoding so one-hot columns stay 0/1.
    if do_scale:
        X = scale_numeric_features(X)

    if do_encode:
        X = encode_categorical(X, drop_first=drop_first)

    if do_split:
        return split_data(
            X,
            y,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    return X, y
