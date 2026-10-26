# P3 — Supply Chain Demand Forecasting with Neural Networks

## Overview

This project explores daily order-volume forecasting for retail supply chains across multiple SKUs. It compares a Random Forest regression baseline with a PyTorch multilayer perceptron (MLP) using batch normalization, dropout, and randomized hyperparameter search.

> **Dataset note:** The dataset in this repository is synthetically generated for demonstration and experimentation. It is not real company or customer data, and results should not be interpreted as production performance.

## Problem Statement

Demand can vary by SKU, category, promotion, region, and shipping conditions. The objective is to use historical order volumes and contextual features to estimate future daily demand and compare model performance on a chronological holdout set.

## Features

- Daily order volume and date-derived features
- SKU category and region
- Promotion discount percentage
- Regional demand index
- Shipping delay days
- Per-SKU lag features (`lag_1`, `lag_7`) and a seven-day rolling mean

Lag and rolling features are shifted so the current target is not included in its own predictors.

## Models and Methodology

1. **Random Forest Regressor:** baseline and comparison model.
2. **PyTorch MLP:** dense feed-forward network with batch normalization and dropout.
3. **Randomized hyperparameter search:** evaluates hidden-layer size, dropout, learning rate, and L2 regularization using validation loss and early stopping.
4. **Chronological holdout evaluation:** final metrics are calculated on the held-out test period, which is not used to select the hyperparameters.

## Repository Structure

```text
P3_Supply_Chain_Demand_Forecasting/
├── models/
│   ├── best_mlp_config.joblib
│   ├── demand_feature_scaler.joblib
│   ├── random_forest_demand.joblib
│   └── tuned_demand_mlp.pt
├── notebook/
│   └── P3_Supply_Chain_Demand_Forecasting.ipynb
└── results/
    ├── p3_daily_demand_overview.png
    ├── p3_evaluation_report.json
    ├── p3_forecasting_dataset.csv
    ├── p3_hyperparameter_search.csv
    ├── p3_mlp_convergence.png
    ├── p3_mlp_validation_loss.png
    ├── p3_model_comparison.csv
    └── p3_supply_chain_demand.csv
```

## Results

The recorded holdout results from the current synthetic-data run are:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest with lag features | 7.9402 | 9.9268 | 0.9019 |
| Tuned PyTorch MLP | 12.3991 | 15.6092 | 0.7575 |

On this run, Random Forest performed better than the tuned MLP on the holdout set. These values are specific to the generated dataset and experiment settings; they are not evidence of real-world business accuracy.

## How to Run

1. Open `notebook/P3_Supply_Chain_Demand_Forecasting.ipynb` in Google Colab or Jupyter.
2. Run the notebook cells in order to generate the synthetic data, prepare lag features, train models, tune the MLP, and evaluate the holdout set.
3. Review the CSV reports and plots under `results/`.

The notebook uses Python, pandas, scikit-learn, matplotlib, and PyTorch. A GPU is optional; the current experiment can run on CPU.

## Deliverables

- Data-generation and forecasting notebook
- Random Forest baseline and tuned PyTorch MLP checkpoint
- Hyperparameter-search results
- Model comparison metrics
- Demand overview and training/validation plots
- JSON evaluation report

## Limitations

- Data is synthetic rather than sourced from a real retailer.
- Model quality may change substantially on real, noisy, operational data.
- This is an educational forecasting workflow, not a production deployment.

## Next Steps

- Validate the pipeline on a real historical retail dataset.
- Add time-series cross-validation and SKU-level error analysis.
- Compare forecasts by category and region, and document prediction intervals.
- Integrate scheduled data ingestion and model monitoring for an operational deployment.
