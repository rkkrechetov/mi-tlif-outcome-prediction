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

## Data

Patient data is not publicly available due to privacy restrictions (ОАО «РЖД» institutional dataset). To reproduce the analysis, replace `../Models/data/features.parquet` and `../Models/data/y.csv` with your own data matching the feature schema described in the paper.

## Citation

> [Citation will be added upon publication]
