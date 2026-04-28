# COMP9417 xRFM Tabular Experiments

This repository contains the code and saved outputs for a COMP9417 project comparing `xRFM` against `XGBoost` and `Random Forest` on tabular supervised learning tasks.

The report only needs a compact experimental-design section. This README records the implementation-level details that are useful for understanding how the experiments were actually run.

## Overview

The project evaluates `xRFM` on five tabular datasets:

- `Adult Income`: binary classification, large mixed-type dataset.
- `AD (Internet Advertisement)`: binary classification, very high-dimensional dataset.
- `Insurance Company Benchmark`: binary classification with 85 features.
- `Bike Sharing`: regression with more than 10,000 samples.
- `Wine Quality`: regression on combined red and white wine data.

These datasets were chosen to cover different sample sizes, feature dimensions, and preprocessing requirements, while also satisfying the project constraints on regression/classification coverage, dataset size, feature count, and mixed feature types.

## Repository Layout

Key directories:

- `experiments/`: dataset-specific loading, tuning, and final evaluation scripts.
- `src/tuning/`: shared tuning utilities for `xRFM`, `XGBoost`, and `Random Forest`.
- `src/utils/`: shared preprocessing, metric, and plotting helpers.
- `outputs/`: saved best parameters, test metrics, AGOP summaries, and plots.
- `notebooks/`: exploratory notebook work.

## Datasets

The saved processed splits in this repository have the following sizes:

| Dataset | Task | Total samples | Processed features |
|---|---|---:|---:|
| Adult Income | Classification | 48,842 | 49 |
| AD | Classification | 1,924 | 1,558 |
| Insurance Company Benchmark | Classification | 9,822 | 85 |
| Bike Sharing | Regression | 17,379 | 12 |
| Wine Quality | Regression | 6,497 | 12 |

## Preprocessing

Preprocessing is handled by dataset-specific scripts under `experiments/*/load_data.py`.

### Shared preprocessing utility

For datasets that use `src/utils/preprocessing.py`, the default pipeline is:

1. remove duplicate rows unless disabled,
2. split features and target,
3. optionally drop rows with missing values before splitting,
4. optionally impute missing values,
5. scale numeric columns with `StandardScaler`,
6. one-hot encode categorical columns with `pd.get_dummies`,
7. split into train, validation, and test sets with `random_state=42`.

The standard split ratio is 60/20/20 because the helper first reserves `val_size + test_size = 0.4`, then splits that equally into validation and test sets.

### Dataset-specific notes

- `Adult Income`
  - Uses the original `adult.data` and `adult.test` files.
  - Several categorical variables are grouped into broader categories before encoding.
  - `fnlwgt` is dropped.
  - Categorical variables are one-hot encoded.
  - Five numeric columns are standardized: `age`, `education_num`, `capital_gain`, `capital_loss`, and `hours_per_week`.
  - The original test file is kept as the final test set.
  - A validation split is taken from the original training portion using a stratified 80/20 split.

- `AD`
  - Loads `ad.data`.
  - Missing values marked with `?` are not imputed; rows containing missing features are dropped before splitting.
  - No scaling is applied.
  - No additional encoding is needed because the remaining columns are already numeric after loading.
  - Uses stratified train/validation/test splitting.

- `Insurance Company Benchmark`
  - Loads the combined benchmark file `TICDATA_TICEVAL_combined.csv`.
  - Metadata columns such as `SOURCE_FILE` and `SOURCE_ROW` are dropped if present.
  - Column names are renamed to more interpretable names.
  - No scaling, encoding, or imputation is applied in the saved pipeline.
  - Uses stratified train/validation/test splitting.

- `Bike Sharing`
  - Loads `hour.csv`.
  - Drops `instant`, `dteday`, `casual`, and `registered`.
  - Uses the shared preprocessing utility with scaling and one-hot encoding enabled.
  - Duplicate removal is disabled.
  - Uses non-stratified train/validation/test splitting because this is a regression task.

- `Wine Quality`
  - Combines red and white wine datasets and adds a categorical `type` feature.
  - Uses the shared preprocessing utility with scaling and one-hot encoding enabled.
  - Duplicate removal is disabled.
  - Uses the same fixed random seed for train/validation/test splitting.

## Models Compared

Every dataset is evaluated with the same three model families:

- `xRFM`
- `XGBoost`
- `Random Forest`

Classification tasks use:

- `xrfm.xRFM`
- `xgboost.XGBClassifier`
- `sklearn.ensemble.RandomForestClassifier`

Regression tasks use:

- `xrfm.xRFM`
- `xgboost.XGBRegressor`
- `sklearn.ensemble.RandomForestRegressor`

## Model Tuning and Selection

All tuning is done on the fixed validation split for each dataset. The test split is reserved for final evaluation.

### xRFM

`xRFM` is mainly tuned through `max_leaf_size`.

- Default shared search space: `256, 512, 1024, 2048`
- Adult uses a larger search space: `2048, 4096, 8192`

### XGBoost

The shared grid for `XGBoost` is:

- `learning_rate`: `0.03, 0.05, 0.1`
- `n_estimators`: `100, 200, 300`
- `max_depth`: `3, 4, 5, 6`
- `reg_alpha`: `0.0, 0.1`
- `reg_lambda`: `1.0, 5.0`

Base settings:

- `subsample=1.0`
- `colsample_bytree=1.0`
- `tree_method="hist"`
- `n_jobs` set by the dataset script

### Random Forest

The shared grid for `Random Forest` is:

- `n_estimators`: `50, 100, 200, 500`
- `max_depth`: `None, 5, 10, 20, 30`
- `min_samples_split`: `2, 5, 10`
- `max_features`: `"sqrt", "log2", None`

### Selection criteria

- Classification models are selected by validation `ROC-AUC` when available, otherwise validation `accuracy`.
- Regression models are selected by validation `RMSE`.

## Final Evaluation Metrics

### Classification

The final evaluation scripts report:

- test `accuracy`
- test `ROC-AUC`
- `training_time_seconds`
- `inference_time_per_sample_seconds`

### Regression

The main reported regression metric is:

- test `RMSE`

Some regression scripts also store:

- `MSE`
- `MAE`
- `R^2`
- training time
- inference time

Note: most scripts record inference cost per sample, but `wine.py` stores total inference time and `insurance_company_benchmark.py` stores both total and per-sample inference time.

## Additional Analyses

Two extra analyses are implemented beyond the basic model comparison:

- `Bike Sharing` includes a scaling experiment with training subsamples of size `1000, 2000, ..., 9000`.
- `Adult`, `Wine Quality`, and `Insurance Company Benchmark` export AGOP-based summaries for `xRFM`.

The Insurance experiment also includes a larger interpretability comparison against:

- mutual information,
- permutation importance from Random Forest,
- PCA on AGOP diagonals,
- feature-rank correlation and top-k overlap summaries.

## Running the Experiments

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate or regenerate dataset splits:

```bash
python3 experiments/adult/load_data.py
python3 experiments/ad/load_data.py
python3 experiments/bike_sharing/load_data.py
python3 experiments/insurance_company_benchmark/load_data.py
python3 experiments/wine_quality/load_data.py
```

Tune models for a dataset:

```bash
python3 experiments/adult/tune_xrfm.py
python3 experiments/adult/tune_xgb.py
python3 experiments/adult/tune_rf.py
```

Run final evaluation for a dataset:

```bash
python3 experiments/adult/adult.py
python3 experiments/ad/ad.py
python3 experiments/bike_sharing/bike_sharing.py
python3 experiments/insurance_company_benchmark/insurance_company_benchmark.py
python3 experiments/wine_quality/wine.py
```

## Saved Outputs

Each dataset has a corresponding folder under `outputs/` containing some combination of:

- `xrfm_results.json`, `xgb_results.json`, `rf_results.json`
- `xrfm_best_params.json`, `xgb_best_params.json`, `rf_best_params.json`
- `test_metrics.json`
- `metrics.csv`
- AGOP summaries such as `xrfm_best_agop.csv`
- interpretability outputs and plots for selected datasets

## Reproducibility Notes

- The code consistently uses `SEED = 42`.
- Validation-based model selection is deterministic given the saved splits and library behavior.
- Some scripts also set environment variables such as `PYTHONHASHSEED`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to stabilize runtime behavior.