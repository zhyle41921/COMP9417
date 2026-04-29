from xgboost import XGBClassifier
from src.tuning.common import (
    evaluate_classification,
    pick_best_classification,
    run_grid_search,
)

SEED = 42

def make_xgb_classifier(params, seed):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        **params,
    )

def tune_xgb(
    X_train,
    y_train,
    X_val,
    y_val,
    results_path,
    best_path,
    seed=SEED,
    base_params=None,
    param_grid=None,
):
    if base_params is None:
        base_params = {"subsample": 1.0, "colsample_bytree": 1.0}

    if param_grid is None:
        param_grid = {
            "learning_rate": [0.03, 0.05, 0.1],
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5, 6],
            "reg_alpha": [0.0, 0.1],
            "reg_lambda": [1.0, 5.0],
        }

    return run_grid_search(
        X_train, y_train, X_val, y_val,
        results_path, best_path, seed, base_params, param_grid,
        make_xgb_classifier, evaluate_classification, pick_best_classification,
    )
