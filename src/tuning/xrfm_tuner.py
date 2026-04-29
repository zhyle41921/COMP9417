from xrfm import xRFM

from src.tuning.common import (
    evaluate_classification,
    pick_best_classification,
    run_grid_search,
)

SEED = 42

def tune_xrfm(
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
        base_params = {"verbose": False}

    if param_grid is None:
        param_grid = {"max_leaf_size": [256, 512, 1024, 2048]}

    def make_model(params, seed):
        return xRFM(**params, random_state=seed)

    return run_grid_search(
        X_train, y_train, X_val, y_val,
        results_path, best_path, seed, base_params, param_grid,
        make_model, evaluate_classification, pick_best_classification,
        fit_kwargs={"X_val": X_val, "y_val": y_val},
    )
