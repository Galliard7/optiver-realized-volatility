# Optiver Realized Volatility Prediction

Predicting short-term realized volatility of stocks from high-frequency order book and trade data. This was a [Kaggle competition](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction) hosted by Optiver in 2021, scored on Root Mean Squared Percentage Error (RMSPE). Notebooks developed on [Kaggle](https://www.kaggle.com/illidan7) using GPU instances.

## Approach

### 1. Exploratory Analysis & Financial Concepts

Introduction to the problem domain: order book mechanics (bid/ask levels, WAP calculation), trade data structure, and realized volatility computation. Establishes the mathematical foundation for feature engineering.

### 2. Feature Engineering — Basic WAP

First-generation features built directly from raw order book snapshots: weighted average price (WAP) from bid/ask levels, log returns, and per-stock realized volatility calculations. Validates that basic book features correlate with the target.

### 3. Feature Engineering — Advanced Order Book

Second-generation features with multi-window aggregations: order book spread, depth imbalance, WAP derivatives across time windows, and 200+ engineered columns. Introduces rolling statistics, moving average crossovers, and Bollinger band-style indicators adapted for tick-level data.

### 4. Stock Clustering

Unsupervised analysis of stock behavior using KMeans and HDBSCAN on volatility profiles. Cluster assignments become categorical features that capture market structure — groups of stocks that move together provide signal beyond individual stock history.

### 5. Dataset Assembly

Merges all feature engineering outputs into unified training/test datasets. Handles time bucket alignment, feature joins across book and trade sources, and produces the final feature matrix used by all downstream models.

### 6. Training — LGBM Baseline

LightGBM model with RMSPE-aware training using custom loss function (1/y^2 sample weights to penalize percentage errors on low-volatility stocks). Establishes the baseline score and identifies which feature groups contribute most.

### 7. Training — LGBM with Bayesian Tuning

Hyperparameter optimization via Bayesian search (Optuna) over learning rate, tree depth, regularization, and feature sampling. Quantifies the improvement ceiling from tuning vs. feature engineering.

### 8. Feature Selection — LGBM

Systematic feature importance analysis and pruning. Removes low-importance and redundant features to reduce overfitting and improve generalization, producing a leaner feature set for the final ensemble.

### 9. Experiment — PyTorch CNN

Alternative approach: raw order book sequences fed directly into a 1D CNN without manual feature engineering. Tests whether a neural network can learn order book patterns end-to-end. Provides diversity signal for ensemble.

### 10. Ensemble — LGBM + NN

Combines 2 LightGBM variants and 2 neural network models. Each model sees different feature subsets or architectures, providing complementary predictions that reduce variance when blended.

### 11. Final Ensemble — Optimized

Production submission with scipy-optimized ensemble weights. Fine-tunes the blending coefficients across the 2 LGBM + 2 NN models to minimize RMSPE on validation folds, producing the final competition submission.

### Feature Engineering Scripts

Standalone Python scripts for production feature pipelines, designed to run as Kaggle utility scripts attached to training/inference notebooks:
- **Order book features** — spread, depth, imbalance, multi-level WAP derivatives
- **Dynamics features** — temporal patterns, moving averages, momentum indicators

## Repository Structure

```
optiver-realized-volatility/
├── README.md
├── .gitignore
├── notebooks/
│   ├── 01-eda-financial-concepts.ipynb             # Order book mechanics and volatility math
│   ├── 02-feature-eng-basic-wap.ipynb              # WAP, log returns, basic realized vol
│   ├── 03-feature-eng-advanced-orderbook.ipynb     # 200+ features: spreads, depth, rolling stats
│   ├── 04-stock-clustering.ipynb                   # KMeans/HDBSCAN on volatility profiles
│   ├── 05-dataset-assembly.ipynb                   # Merge all features into training matrix
│   ├── 06-training-lgbm-baseline.ipynb             # LGBM with RMSPE custom loss (1/y² weights)
│   ├── 07-training-lgbm-bayesian-tuning.ipynb      # Optuna hyperparameter optimization
│   ├── 08-feature-selection-lgbm.ipynb             # Importance-based feature pruning
│   ├── 09-experiment-pytorch-cnn.ipynb             # 1D CNN on raw order book sequences
│   ├── 10-ensemble-lgbm-nn.ipynb                   # 2 LGBM + 2 NN blending
│   └── 11-final-ensemble-optimized.ipynb           # Scipy-optimized ensemble weights
└── src/
    ├── feateng-orderbook.py                        # Order book feature pipeline (utility script)
    └── feateng-dynamics.py                         # Temporal dynamics feature pipeline
```

## Tech Stack

- **ML**: LightGBM, PyTorch, scikit-learn, Optuna
- **Data**: pandas, NumPy
- **Clustering**: KMeans, HDBSCAN
- **Optimization**: SciPy (ensemble weight tuning)

## Competition

| | |
|---|---|
| **Competition** | [Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction) |
| **Type** | Featured prediction (tabular) |
| **Metric** | RMSPE (Root Mean Squared Percentage Error) |
| **Timeline** | June 2021 -- September 2021 |
