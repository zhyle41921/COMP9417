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
    stratify_labels = y if stratify else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=val_size + test_size,
        stratify=stratify_labels,
        random_state=random_state
    )

    test_ratio = test_size / (val_size + test_size)
    stratify_temp = y_temp if stratify else None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_ratio,
        stratify=stratify_temp,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def impute_missing_values(X_train, X_val=None, X_test=None):
    X_train = X_train.copy()
    X_val = None if X_val is None else X_val.copy()
    X_test = None if X_test is None else X_test.copy()

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    for col in categorical_cols:
        fill_value = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(fill_value)
        if X_val is not None:
            X_val[col] = X_val[col].fillna(fill_value)
        if X_test is not None:
            X_test[col] = X_test[col].fillna(fill_value)

    for col in numeric_cols:
        fill_value = X_train[col].mean()
        X_train[col] = X_train[col].fillna(fill_value)
        if X_val is not None:
            X_val[col] = X_val[col].fillna(fill_value)
        if X_test is not None:
            X_test[col] = X_test[col].fillna(fill_value)

    return X_train, X_val, X_test


def encode_categorical(X_train, X_val=None, X_test=None, drop_first=True):
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns

    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=drop_first)

    if X_val is not None:
        X_val = pd.get_dummies(X_val, columns=categorical_cols, drop_first=drop_first)
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    if X_test is not None:
        X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=drop_first)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train, X_val, X_test


def scale_features(X_train, X_val=None, X_test=None):
    X_train = X_train.copy()
    X_val = None if X_val is None else X_val.copy()
    X_test = None if X_test is None else X_test.copy()

    numeric_cols = X_train.select_dtypes(exclude=["object", "category"]).columns
    scaler = StandardScaler()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

    if X_val is not None:
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    if X_test is not None:
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_val, X_test


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
):
    df = df.copy()

    if do_remove_duplicates:
        df = remove_duplicates(df)

    X, y = split_features_target(df, target_col)

    if do_split:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X,
            y,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
    else:
        X_train, y_train = X, y
        X_val, X_test, y_val, y_test = None, None, None, None

    if do_impute:
        X_train, X_val, X_test = impute_missing_values(X_train, X_val, X_test)

    if do_encode:
        X_train, X_val, X_test = encode_categorical(X_train, X_val, X_test)

    if do_scale:
        X_train, X_val, X_test = scale_features(X_train, X_val, X_test)

    if do_split:
        return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, y_train