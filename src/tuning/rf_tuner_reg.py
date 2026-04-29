from sklearn.ensemble import RandomForestRegressor

from src.tuning.common import evaluate_regression, pick_best_regression, run_grid_search

SEED = 42


def tune_rf(
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
        base_params = {}

    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2", None],
        }

    def make_model(params, seed):
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **params)

    return run_grid_search(
        X_train, y_train, X_val, y_val,
        results_path, best_path, seed, base_params, param_grid,
        make_model, evaluate_regression, pick_best_regression,
    )
