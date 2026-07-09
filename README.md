# Predicting Outcomes of Minimally Invasive Lumbar Fusion: Classic ML Regression

Regression models for predicting 3-year functional outcomes after minimally invasive lumbar spinal fusion in railway workers, based on preoperative clinical, radiological, and neuroimaging parameters.

## Overview

**Task:** regression (3 independent targets)
**Targets:** ODI (Oswestry Disability Index), SF-36 PCS (Physical Component Summary), SF-36 MCS (Mental Component Summary)
**Sample:** n = 457 patients, prospective 3-year follow-up
**Features:** 27 preoperative variables (19 clinical/radiological/MRI + 8 engineered features)

## Models

| Model | Type |
|---|---|
| Ridge | L2-regularized linear regression |
| Lasso | L1-regularized linear regression |
| ElasticNet | L1/L2 hybrid |
| XGBoost | Gradient boosting |
| LightGBM | Gradient boosting |
| CatBoost | Gradient boosting |

## Validation

Nested cross-validation:
- **Outer loop:** LOOCV (457 iterations) — unbiased out-of-sample predictions
- **Inner loop:** Optuna TPE + 5-fold CV — hyperparameter search per fold
- **Preprocessing:** StandardScaler + OneHotEncoder fit on training fold only (no leakage)

Model performance assessed via the **ABCD framework** (Steyerberg & Vergouwe, 2014):
- **A** — Calibration-in-the-large (bias)
- **B** — Calibration slope (β₀, β₁)
- **C** — Discrimination (MAE, RMSE, R²)
- **D** — Clinical utility (% predictions within MCID)

MCID thresholds: ODI = 10 pts, SF-36 PCS/MCS = 5 pts.

## Results

| Target | Best model | MAE | RMSE | R² | % MCID |
|---|---|---|---|---|---|
| ODI | XGBoost | 4.48 | 6.00 | 0.509 | 90.4% |
| SF-36 PCS | CatBoost | 5.34 | 6.87 | 0.557 | 55.4% |
| SF-36 MCS | XGBoost | 5.98 | 7.95 | 0.405 | 52.5% |

Gradient boosting models outperformed linear regressors across all three targets. Feature importance (SHAP) consistently ranked ADC and ADC/Pfirrmann ratio as the top predictors.

## Model artifacts

Final models trained on the full cohort (n = 457) are available in `res/`:

| File | Description |
|---|---|
| `res/final_models.pkl` | Fitted models (Ridge, Lasso, ElasticNet, XGBoost, LightGBM, CatBoost) for ODI, SF-36 PCS, SF-36 MCS |
| `res/final_params.pkl` | Optimal hyperparameters per model |
| `res/final_preprocessor.pkl` | Preprocessing pipeline (StandardScaler + OneHotEncoder), fit on full sample |

Load with `joblib.load()` to generate predictions on new preoperative data.

Repository: [github.com/rkkrechetov/spine-fusion-outcome-classic-ml](https://github.com/rkkrechetov/spine-fusion-outcome-classic-ml)

## Notebooks

| Notebook | Description |
|---|---|
| `01-eda.ipynb` | Descriptive statistics tables (categorical + numerical) |
| `02-learning-models.ipynb` | LOOCV training, Optuna hyperparameter search, final model fitting |
| `03-validating-models.ipynb` | ABCD evaluation, calibration plots, SHAP beeswarm plots |

## Requirements

```
numpy
pandas
scikit-learn
xgboost
lightgbm
catboost
optuna
shap
matplotlib
scipy
joblib
tqdm
pyarrow
```

Install:

```bash
pip install -r requirements.txt
```

## Citation

> [Citation will be added upon publication]
