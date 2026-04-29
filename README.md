# COMP9417 xRFM project

This is the code for our COMP9417 group project. We compare `xRFM` with `XGBoost`
and `Random Forest` on a few tabular datasets, using both classification and
regression tasks.

The repo is mainly scripts, not a package or app. There is no compile step as
such, just install the Python requirements and run the scripts from the project
root.

## Folder layout

- `experiments/` has one folder per dataset.
- `src/tuning/` has the shared tuning code for the three model types.
- `src/utils/` has the small preprocessing / metrics / plotting helpers.
- `outputs/` has the saved parameters and results from runs we already did.
- `notebooks/` has the rough notebook work.

## Datasets used

We used five datasets:

| Dataset | Task | Rows | Features after processing |
|---|---:|---:|---:|
| Adult Income | classification | 48,842 | 49 |
| Internet Advertisement | classification | 1,924 | 1,558 |
| Insurance Company Benchmark | classification | 9,822 | 85 |
| Bike Sharing | regression | 17,379 | 12 |
| Wine Quality | regression | 6,497 | 12 |

The main reason for using these was to get a mix of dataset sizes, regression vs
classification, and easy vs annoying preprocessing.

## Setup

Use Python 3.10+ if possible. A virtual environment is recommended so the
packages do not mess with your global Python install.

```bash
cd COMP9417
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already have an environment set up, the only important command is:

```bash
pip install -r requirements.txt
```

The main packages are `numpy`, `pandas`, `scikit-learn`, `xgboost`, `xrfm`,
`torch`, `matplotlib`, and `seaborn`.

## How to run it

Run commands from the repo root, i.e. the folder containing this README.

### 1. Rebuild the processed data splits

The data split scripts create the train/validation/test CSV files used by the
later scripts.

```bash
python3 experiments/adult/load_data.py
python3 experiments/ad/load_data.py
python3 experiments/bike_sharing/load_data.py
python3 experiments/insurance_company_benchmark/load_data.py
python3 experiments/wine_quality/load_data.py
```

You do not need to run these every time if the processed CSVs are already there.

### 2. Tune models

Each dataset has three tuning scripts, one for each model. For example, for the
Adult dataset:

```bash
python3 experiments/adult/tune_xrfm.py
python3 experiments/adult/tune_xgb.py
python3 experiments/adult/tune_rf.py
```

Same idea for the other datasets:

```bash
python3 experiments/ad/tune_xrfm.py
python3 experiments/ad/tune_xgb.py
python3 experiments/ad/tune_rf.py

python3 experiments/bike_sharing/tune_xrfm.py
python3 experiments/bike_sharing/tune_xgb.py
python3 experiments/bike_sharing/tune_rf.py

python3 experiments/insurance_company_benchmark/tune_xrfm.py
python3 experiments/insurance_company_benchmark/tune_xgb.py
python3 experiments/insurance_company_benchmark/tune_rf.py

python3 experiments/wine_quality/tune_xrfm.py
python3 experiments/wine_quality/tune_xgb.py
python3 experiments/wine_quality/tune_rf.py
```

Some of these take a while, especially the xRFM runs on bigger datasets.

### 3. Run final evaluation

These scripts load the best saved params and write the final metrics into
`outputs/`.

```bash
python3 experiments/adult/adult.py
python3 experiments/ad/ad.py
python3 experiments/bike_sharing/bike_sharing.py
python3 experiments/insurance_company_benchmark/insurance_company_benchmark.py
python3 experiments/wine_quality/wine.py
```

`bike_sharing.py` also runs the smaller scaling experiment, so it prints and
saves more rows than the other scripts.

## What the scripts output

The results are saved under `outputs/<dataset>/`. The exact files depend on the
dataset, but the common ones are:

- `*_best_params.json` for tuned hyperparameters.
- `*_results.json` for validation/tuning results.
- `metrics.csv` for the final model comparison.
- `test_metrics.json` for final test-set values.
- `xrfm_best_agop.csv` and related files for the AGOP summaries.

The Insurance experiment also writes extra interpretability files under
`outputs/insurance_company_benchmark/interpretability/`.

## Model details

For classification we used:

- `xrfm.xRFM`
- `xgboost.XGBClassifier`
- `sklearn.ensemble.RandomForestClassifier`

For regression we used:

- `xrfm.xRFM`
- `xgboost.XGBRegressor`
- `sklearn.ensemble.RandomForestRegressor`

The validation split was used for selecting parameters. The test split was kept
for final evaluation. Classification uses accuracy and ROC-AUC. Regression uses
RMSE, plus some scripts save extra values like MAE or `R^2`.

## Randomness

Most scripts use `SEED = 42`. A few scripts also set thread-related environment
variables so timing is less all over the place, but exact runtimes can still
change depending on the machine.