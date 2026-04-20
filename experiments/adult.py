import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from src.utils.preprocessing import preprocess_data
from xgboost import XGBClassifier, XGBRegressor
from xrfm import xRFM

columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

df = pd.read_csv(
    "../datasets/adult.data",
    header=None,
    names=columns,
    skipinitialspace=True,
    na_values="?"
)

X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df, target_col="income", do_remove_duplicates=False)

print(X_train.shape, y_train.shape)