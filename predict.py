"""
predict.py
Loads the trained model and predicts house prices for new/unseen houses.
Run with: python predict.py
"""

import pandas as pd
import joblib

model = joblib.load("models/house_price_model.pkl")

# Example: a few new houses to predict prices for
new_houses = pd.DataFrame([
    {
        "OverallQual": 8, "GrLivArea": 2400, "GarageCars": 2,
        "TotalBsmtSF": 1200, "FullBath": 2, "YearBuilt": 2015,
        "LotArea": 9000, "Neighborhood": "Downtown",
        "HouseStyle": "2Story", "CentralAir": "Y",
    },
    {
        "OverallQual": 4, "GrLivArea": 1100, "GarageCars": 1,
        "TotalBsmtSF": 400, "FullBath": 1, "YearBuilt": 1965,
        "LotArea": 5000, "Neighborhood": "Rural",
        "HouseStyle": "1Story", "CentralAir": "N",
    },
])

predictions = model.predict(new_houses)

for i, price in enumerate(predictions):
    print(f"House {i + 1}: Predicted Sale Price = ${price:,.0f}")
