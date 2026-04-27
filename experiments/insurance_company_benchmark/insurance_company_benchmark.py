SEED = 42
N_THREADS = 4

import os

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_THREADS)

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from xrfm import xRFM
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from experiments.insurance_company_benchmark.load_data import load_insurance_splits


def load_best_params():
    output_dir = ROOT / "outputs" / "insurance_company_benchmark"

    with open(output_dir / "xrfm_best_params.json", "r") as f:
        xrfm_result = json.load(f)

    with open(output_dir / "xgb_best_params.json", "r") as f:
        xgb_result = json.load(f)

    with open(output_dir / "rf_best_params.json", "r") as f:
        rf_result = json.load(f)

    return xrfm_result["params"], xgb_result["params"], rf_result["params"]


def evaluate_model(model, X, y):
    start = time.perf_counter()
    y_pred = model.predict(X)
    inference_time = time.perf_counter() - start

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "inference_time_seconds": float(inference_time),
        "inference_time_per_sample_seconds": float(inference_time / len(y)),
    }

    if hasattr(model, "predict_proba"):
        y_score = np.asarray(model.predict_proba(X))

        if y_score.ndim == 2 and y_score.shape[1] >= 2:
            y_score = y_score[:, 1]
        elif y_score.ndim == 2 and y_score.shape[1] == 1:
            y_score = y_score[:, 0]

        metrics["roc_auc"] = float(roc_auc_score(y, y_score))

    return metrics


def fit_with_time(model, *fit_args, **fit_kwargs):
    start = time.perf_counter()
    model.fit(*fit_args, **fit_kwargs)
    training_time = time.perf_counter() - start

    return model, float(training_time)


def to_numpy_agop(agop):
    if hasattr(agop, "detach"):
        agop = agop.detach().cpu().numpy()
    else:
        agop = np.asarray(agop)

    if agop.ndim == 1:
        agop = np.diag(agop)

    return agop


def rank_features(df, score_col, rank_col):
    df = df.copy()
    df[rank_col] = df[score_col].rank(ascending=False, method="min")
    return df


def extract_agop_diagonals(model, feature_names, interpret_dir):
    agops = model.collect_best_agops()

    if len(agops) == 0:
        raise ValueError("No AGOP matrices found.")

    rows = []
    full_agop_scores = []

    for leaf_idx, agop in enumerate(agops):
        agop = to_numpy_agop(agop)

        diag = np.diag(agop)
        eigvals = np.linalg.eigvalsh(agop)

        full_agop_scores.append({
            "leaf_index": leaf_idx,
            "max_eigenvalue": float(np.max(eigvals)),
            "trace": float(np.trace(agop)),
            "frobenius_norm": float(np.linalg.norm(agop, ord="fro")),
        })

        for feature, value in zip(feature_names, diag):
            rows.append({
                "leaf_index": leaf_idx,
                "feature": feature,
                "agop_diagonal": float(value),
                "abs_agop_diagonal": float(abs(value)),
            })

    leaf_diag_df = pd.DataFrame(rows)
    leaf_score_df = pd.DataFrame(full_agop_scores)

    leaf_diag_df.to_csv(
        interpret_dir / "xrfm_agop_diagonal_by_leaf.csv",
        index=False,
    )

    leaf_score_df.to_csv(
        interpret_dir / "xrfm_agop_leaf_matrix_scores.csv",
        index=False,
    )

    avg_diag_df = (
        leaf_diag_df
        .groupby("feature", as_index=False)
        .agg(
            mean_agop_diagonal=("agop_diagonal", "mean"),
            mean_abs_agop_diagonal=("abs_agop_diagonal", "mean"),
            max_abs_agop_diagonal=("abs_agop_diagonal", "max"),
            std_abs_agop_diagonal=("abs_agop_diagonal", "std"),
        )
        .sort_values("mean_abs_agop_diagonal", ascending=False)
    )

    avg_diag_df = rank_features(
        avg_diag_df,
        score_col="mean_abs_agop_diagonal",
        rank_col="agop_rank",
    )

    avg_diag_df.to_csv(
        interpret_dir / "xrfm_agop_diagonal_summary.csv",
        index=False,
    )

    return leaf_diag_df, leaf_score_df, avg_diag_df


def compute_agop_pca_loadings(leaf_diag_df, feature_names, interpret_dir, n_components=5):
    agop_matrix = (
        leaf_diag_df
        .pivot(index="leaf_index", columns="feature", values="abs_agop_diagonal")
        .reindex(columns=feature_names)
        .fillna(0.0)
    )

    max_components = min(n_components, agop_matrix.shape[0], agop_matrix.shape[1])

    if max_components < 1:
        raise ValueError("Not enough AGOP diagonal data for PCA.")

    scaler = StandardScaler()
    Z = scaler.fit_transform(agop_matrix)

    pca = PCA(n_components=max_components, random_state=SEED)
    pca.fit(Z)

    loading_rows = []

    for pc_idx in range(max_components):
        for feature, loading in zip(feature_names, pca.components_[pc_idx]):
            loading_rows.append({
                "component": f"PC{pc_idx + 1}",
                "feature": feature,
                "loading": float(loading),
                "abs_loading": float(abs(loading)),
                "explained_variance_ratio": float(
                    pca.explained_variance_ratio_[pc_idx]
                ),
            })

    pca_loadings_df = pd.DataFrame(loading_rows)

    pca_loadings_df.to_csv(
        interpret_dir / "agop_pca_loadings.csv",
        index=False,
    )

    explained_df = pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(max_components)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(
            pca.explained_variance_ratio_
        ),
    })

    explained_df.to_csv(
        interpret_dir / "agop_pca_explained_variance.csv",
        index=False,
    )

    pc1_df = (
        pca_loadings_df[pca_loadings_df["component"] == "PC1"]
        .copy()
        .sort_values("abs_loading", ascending=False)
    )

    pc1_df = rank_features(
        pc1_df,
        score_col="abs_loading",
        rank_col="pca_pc1_rank",
    )

    pc1_df.to_csv(
        interpret_dir / "agop_pca_pc1_feature_ranking.csv",
        index=False,
    )

    return pca_loadings_df, explained_df, pc1_df


def compute_mutual_information(X_train_df, y_train_s, interpret_dir):
    mi = mutual_info_classif(
        X_train_df,
        y_train_s,
        discrete_features="auto",
        random_state=SEED,
        n_neighbors=3,
    )

    mi_df = pd.DataFrame({
        "feature": X_train_df.columns,
        "mutual_information": mi,
    }).sort_values("mutual_information", ascending=False)

    mi_df = rank_features(
        mi_df,
        score_col="mutual_information",
        rank_col="mi_rank",
    )

    mi_df.to_csv(
        interpret_dir / "mutual_information_scores.csv",
        index=False,
    )

    return mi_df


def compute_permutation_importance_scores(model, X_test_df, y_test_s, interpret_dir):
    result = permutation_importance(
        model,
        X_test_df,
        y_test_s,
        scoring="roc_auc",
        n_repeats=20,
        random_state=SEED,
        n_jobs=N_THREADS,
    )

    perm_df = pd.DataFrame({
        "feature": X_test_df.columns,
        "permutation_importance_mean": result.importances_mean,
        "permutation_importance_std": result.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False)

    perm_df = rank_features(
        perm_df,
        score_col="permutation_importance_mean",
        rank_col="permutation_rank",
    )

    perm_df.to_csv(
        interpret_dir / "permutation_importance_scores.csv",
        index=False,
    )

    return perm_df


def make_interpretability_comparison(
    agop_summary_df,
    pc1_df,
    mi_df,
    perm_df,
    interpret_dir,
    top_k=20,
):
    comparison_df = (
        agop_summary_df[[
            "feature",
            "mean_abs_agop_diagonal",
            "max_abs_agop_diagonal",
            "std_abs_agop_diagonal",
            "agop_rank",
        ]]
        .merge(
            pc1_df[["feature", "abs_loading", "pca_pc1_rank"]],
            on="feature",
            how="left",
        )
        .merge(
            mi_df[["feature", "mutual_information", "mi_rank"]],
            on="feature",
            how="left",
        )
        .merge(
            perm_df[[
                "feature",
                "permutation_importance_mean",
                "permutation_importance_std",
                "permutation_rank",
            ]],
            on="feature",
            how="left",
        )
    )

    comparison_df["mean_rank"] = comparison_df[[
        "agop_rank",
        "pca_pc1_rank",
        "mi_rank",
        "permutation_rank",
    ]].mean(axis=1)

    comparison_df["rank_range"] = comparison_df[[
        "agop_rank",
        "pca_pc1_rank",
        "mi_rank",
        "permutation_rank",
    ]].max(axis=1) - comparison_df[[
        "agop_rank",
        "pca_pc1_rank",
        "mi_rank",
        "permutation_rank",
    ]].min(axis=1)

    comparison_df = comparison_df.sort_values("mean_rank")

    comparison_df.to_csv(
        interpret_dir / "interpretability_feature_comparison.csv",
        index=False,
    )

    rank_cols = [
        "agop_rank",
        "pca_pc1_rank",
        "mi_rank",
        "permutation_rank",
    ]

    rank_corr_df = comparison_df[rank_cols].corr(method="spearman")

    rank_corr_df.to_csv(
        interpret_dir / "interpretability_rank_spearman_correlation.csv",
    )

    score_cols = [
        "mean_abs_agop_diagonal",
        "abs_loading",
        "mutual_information",
        "permutation_importance_mean",
    ]

    score_corr_df = comparison_df[score_cols].corr(method="spearman")

    score_corr_df.to_csv(
        interpret_dir / "interpretability_score_spearman_correlation.csv",
    )

    top_sets = {
        "agop": set(
            comparison_df
            .sort_values("agop_rank")
            .head(top_k)["feature"]
        ),
        "pca_pc1": set(
            comparison_df
            .sort_values("pca_pc1_rank")
            .head(top_k)["feature"]
        ),
        "mutual_information": set(
            comparison_df
            .sort_values("mi_rank")
            .head(top_k)["feature"]
        ),
        "permutation_importance": set(
            comparison_df
            .sort_values("permutation_rank")
            .head(top_k)["feature"]
        ),
    }

    overlap_rows = []

    methods = list(top_sets.keys())

    for i, method_a in enumerate(methods):
        for method_b in methods[i + 1:]:
            overlap = top_sets[method_a].intersection(top_sets[method_b])
            union = top_sets[method_a].union(top_sets[method_b])

            overlap_rows.append({
                "method_a": method_a,
                "method_b": method_b,
                "top_k": top_k,
                "overlap_count": len(overlap),
                "jaccard_similarity": len(overlap) / len(union),
                "overlapping_features": ", ".join(sorted(overlap)),
            })

    overlap_df = pd.DataFrame(overlap_rows)

    overlap_df.to_csv(
        interpret_dir / f"interpretability_top_{top_k}_overlap.csv",
        index=False,
    )

    top_features_df = comparison_df.head(top_k).copy()

    top_features_df.to_csv(
        interpret_dir / f"interpretability_top_{top_k}_combined_features.csv",
        index=False,
    )

    return comparison_df, rank_corr_df, score_corr_df, overlap_df


def run_interpretability_analysis(
    xrfm_model,
    rf_model,
    X_train_df,
    X_test_df,
    y_train_s,
    y_test_s,
    output_dir,
    top_k=20,
):
    interpret_dir = output_dir / "interpretability"
    interpret_dir.mkdir(parents=True, exist_ok=True)

    feature_names = list(X_train_df.columns)

    leaf_diag_df, leaf_score_df, agop_summary_df = extract_agop_diagonals(
        model=xrfm_model,
        feature_names=feature_names,
        interpret_dir=interpret_dir,
    )

    pca_loadings_df, pca_explained_df, pc1_df = compute_agop_pca_loadings(
        leaf_diag_df=leaf_diag_df,
        feature_names=feature_names,
        interpret_dir=interpret_dir,
        n_components=5,
    )

    mi_df = compute_mutual_information(
        X_train_df=X_train_df,
        y_train_s=y_train_s,
        interpret_dir=interpret_dir,
    )

    perm_df = compute_permutation_importance_scores(
        model=rf_model,
        X_test_df=X_test_df,
        y_test_s=y_test_s,
        interpret_dir=interpret_dir,
    )

    comparison_df, rank_corr_df, score_corr_df, overlap_df = (
        make_interpretability_comparison(
            agop_summary_df=agop_summary_df,
            pc1_df=pc1_df,
            mi_df=mi_df,
            perm_df=perm_df,
            interpret_dir=interpret_dir,
            top_k=top_k,
        )
    )

    summary = {
        "files_created": [
            "xrfm_agop_diagonal_by_leaf.csv",
            "xrfm_agop_leaf_matrix_scores.csv",
            "xrfm_agop_diagonal_summary.csv",
            "agop_pca_loadings.csv",
            "agop_pca_explained_variance.csv",
            "agop_pca_pc1_feature_ranking.csv",
            "mutual_information_scores.csv",
            "permutation_importance_scores.csv",
            "interpretability_feature_comparison.csv",
            "interpretability_rank_spearman_correlation.csv",
            "interpretability_score_spearman_correlation.csv",
            f"interpretability_top_{top_k}_overlap.csv",
            f"interpretability_top_{top_k}_combined_features.csv",
        ],
        "top_k": top_k,
        "n_features": len(feature_names),
        "n_agop_leaves": int(leaf_diag_df["leaf_index"].nunique()),
    }

    with open(interpret_dir / "interpretability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "interpret_dir": interpret_dir,
        "summary": summary,
        "comparison_df": comparison_df,
        "rank_corr_df": rank_corr_df,
        "score_corr_df": score_corr_df,
        "overlap_df": overlap_df,
    }


def make_metrics_csv(xrfm_metrics, xgb_metrics, rf_metrics, output_dir):
    metrics_df = pd.DataFrame([
        {
            "model": "xrfm",
            "accuracy": xrfm_metrics["accuracy"],
            "roc_auc": xrfm_metrics.get("roc_auc", ""),
            "training_time_seconds": xrfm_metrics["training_time_seconds"],
            "inference_time_seconds": xrfm_metrics["inference_time_seconds"],
            "inference_time_per_sample_seconds": xrfm_metrics[
                "inference_time_per_sample_seconds"
            ],
        },
        {
            "model": "xgboost",
            "accuracy": xgb_metrics["accuracy"],
            "roc_auc": xgb_metrics.get("roc_auc", ""),
            "training_time_seconds": xgb_metrics["training_time_seconds"],
            "inference_time_seconds": xgb_metrics["inference_time_seconds"],
            "inference_time_per_sample_seconds": xgb_metrics[
                "inference_time_per_sample_seconds"
            ],
        },
        {
            "model": "random_forest",
            "accuracy": rf_metrics["accuracy"],
            "roc_auc": rf_metrics.get("roc_auc", ""),
            "training_time_seconds": rf_metrics["training_time_seconds"],
            "inference_time_seconds": rf_metrics["inference_time_seconds"],
            "inference_time_per_sample_seconds": rf_metrics[
                "inference_time_per_sample_seconds"
            ],
        },
    ])

    metrics_csv_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    return metrics_df, metrics_csv_path


def main():
    output_dir = ROOT / "outputs" / "insurance_company_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s = (
        load_insurance_splits()
    )

    best_xrfm_params, best_xgb_params, best_rf_params = load_best_params()

    X_train_np = np.asarray(X_train_df, dtype=np.float32)
    X_val_np = np.asarray(X_val_df, dtype=np.float32)
    X_test_np = np.asarray(X_test_df, dtype=np.float32)

    y_train_np = np.asarray(y_train_s, dtype=np.int64)
    y_val_np = np.asarray(y_val_s, dtype=np.int64)
    y_test_np = np.asarray(y_test_s, dtype=np.int64)

    xrfm_model = xRFM(
        **best_xrfm_params,
        n_threads=N_THREADS,
        random_state=SEED,
    )

    xrfm_model, xrfm_training_time = fit_with_time(
        xrfm_model,
        X_train_np,
        y_train_np,
        X_val=X_val_np,
        y_val=y_val_np,
    )

    xrfm_metrics = evaluate_model(xrfm_model, X_test_np, y_test_np)
    xrfm_metrics["training_time_seconds"] = xrfm_training_time

    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=N_THREADS,
        **best_xgb_params,
    )

    xgb_model, xgb_training_time = fit_with_time(
        xgb_model,
        X_train_df,
        y_train_s,
    )

    xgb_metrics = evaluate_model(xgb_model, X_test_df, y_test_s)
    xgb_metrics["training_time_seconds"] = xgb_training_time

    rf_model = RandomForestClassifier(
        random_state=SEED,
        n_jobs=N_THREADS,
        **best_rf_params,
    )

    rf_model, rf_training_time = fit_with_time(
        rf_model,
        X_train_df,
        y_train_s,
    )

    rf_metrics = evaluate_model(rf_model, X_test_df, y_test_s)
    rf_metrics["training_time_seconds"] = rf_training_time

    interpretability_results = run_interpretability_analysis(
        xrfm_model=xrfm_model,
        rf_model=rf_model,
        X_train_df=X_train_df,
        X_test_df=X_test_df,
        y_train_s=y_train_s,
        y_test_s=y_test_s,
        output_dir=output_dir,
        top_k=20,
    )

    results = {
        "xrfm": {
            "params": best_xrfm_params,
            "test_metrics": xrfm_metrics,
        },
        "xgboost": {
            "params": best_xgb_params,
            "test_metrics": xgb_metrics,
        },
        "random_forest": {
            "params": best_rf_params,
            "test_metrics": rf_metrics,
        },
        "interpretability": {
            "output_dir": str(interpretability_results["interpret_dir"]),
            "summary": interpretability_results["summary"],
        },
    }

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    metrics_df, metrics_csv_path = make_metrics_csv(
        xrfm_metrics=xrfm_metrics,
        xgb_metrics=xgb_metrics,
        rf_metrics=rf_metrics,
        output_dir=output_dir,
    )

    print("\nTest metrics:")
    print(json.dumps(results, indent=2))

    print("\nSaved metrics to:")
    print(metrics_csv_path)

    print("\nMetrics:")
    print(metrics_df)

    print("\nInterpretability outputs saved to:")
    print(interpretability_results["interpret_dir"])

    print("\nTop combined interpretability features:")
    print(interpretability_results["comparison_df"].head(20))

    print("\nRank agreement:")
    print(interpretability_results["rank_corr_df"])

    print("\nTop-20 overlap:")
    print(interpretability_results["overlap_df"])


if __name__ == "__main__":
    main()