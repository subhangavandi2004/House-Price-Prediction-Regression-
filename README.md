# House Price Prediction (Regression)

Machine learning regression project for predicting house prices using Python, Pandas, Scikit-learn, and feature engineering.

## Overview

- **Problem**: Predict `SalePrice` (a continuous value) from house features.
- **Approach**: EDA-informed preprocessing (scaling + one-hot encoding) →
  compared a Linear Regression baseline against a Random Forest Regressor →
  selected the best-performing model automatically based on R².
- **Result**: R² ≈ 0.97, MAE ≈ $11,500 on held-out test data.

## Tech Stack
Python · Pandas · NumPy · Scikit-learn

## Project Structure
```
house-price-prediction/
├── data/
│   └── house_prices.csv         # dataset
├── models/
│   └── house_price_model.pkl    # trained pipeline (best of 2 models)
├── generate_data.py             # generates the dataset
├── train_model.py               # trains, compares, and evaluates models
├── predict.py                   # example: load model, predict new houses
├── requirements.txt
└── README.md
```

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Generate the dataset:
   ```
   python generate_data.py
   ```

3. Train and evaluate the models:
   ```
   python train_model.py
   ```
   This prints metrics (MAE, RMSE, R²) for both Linear Regression and
   Random Forest, plus the top 5 most important features, then saves
   whichever model performed better.

4. Predict prices for new houses:
   ```
   python predict.py
   ```

## Model Comparison

| Model             | MAE     | RMSE    | R²   |
|-------------------|---------|---------|------|
| Linear Regression | 11,505  | 14,662  | 0.971|
| Random Forest     | 18,494  | 23,799  | 0.924|

**Top predictive features**: Above-ground living area (`GrLivArea`) and
overall quality rating (`OverallQual`) drive most of the price variation,
followed by basement size and lot area.

## Note on the Dataset
This project uses a synthetically generated dataset that mirrors the
structure of the well-known Kaggle "House Prices — Advanced Regression
Techniques" dataset, so the full pipeline can be reproduced without any
external download. The same code works unchanged on the real Kaggle
dataset — just replace `data/house_prices.csv` and adjust the column
names in `train_model.py` if needed.

## Author
Ahamad Subhan Abubakar Gavandi
