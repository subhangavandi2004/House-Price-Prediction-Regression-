"""
train_model.py
Loads the house price dataset, does EDA-informed preprocessing, trains a
RandomForestRegressor (with a LinearRegression baseline for comparison),
evaluates both, and saves the best model to models/house_price_model.pkl.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("data/house_prices.csv")

X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n=== {name} ===")
    print(f"MAE : {mae:,.0f}")
    print(f"RMSE: {rmse:,.0f}")
    print(f"R^2 : {r2:.3f}")
    return r2


# --- Baseline: Linear Regression ---
lin_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression()),
])
lin_model.fit(X_train, y_train)
lin_r2 = evaluate("Linear Regression (baseline)", lin_model, X_test, y_test)

# --- Random Forest Regressor ---
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300, max_depth=12, random_state=42
    )),
])
rf_model.fit(X_train, y_train)
rf_r2 = evaluate("Random Forest Regressor", rf_model, X_test, y_test)

# --- Feature importance (Random Forest) ---
feature_names = (
    numeric_cols +
    list(rf_model.named_steps["preprocessor"]
         .named_transformers_["cat"].get_feature_names_out(categorical_cols))
)
importances = rf_model.named_steps["regressor"].feature_importances_
top_features = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:5]
print("\nTop 5 most important features:")
for name, imp in top_features:
    print(f"  {name}: {imp:.3f}")

# --- Save the better model ---
best_model, best_name = (rf_model, "Random Forest") if rf_r2 > lin_r2 else (lin_model, "Linear Regression")
joblib.dump(best_model, "models/house_price_model.pkl")
print(f"\nBest model ({best_name}) saved -> models/house_price_model.pkl")
