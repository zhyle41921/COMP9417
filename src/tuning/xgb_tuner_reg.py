from xgboost import XGBRegressor

from src.tuning.common import evaluate_regression, pick_best_regression, run_grid_search

SEED = 42


def make_xgb_regressor(params, seed):
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        **params,
    )


def tune_xgb_regression(
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
        make_xgb_regressor, evaluate_regression, pick_best_regression,
    )
