"""
generate_data.py
Creates a synthetic house price dataset (mirrors the structure of the
popular Kaggle 'House Prices' dataset) so the project runs end-to-end
without needing an external download.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1500

overall_qual = np.random.randint(1, 11, N)             # 1-10 quality rating
gr_liv_area = np.random.randint(500, 4500, N)           # sq ft living area
garage_cars = np.random.randint(0, 4, N)
total_bsmt_sf = np.random.randint(0, 2500, N)
full_bath = np.random.randint(1, 4, N)
year_built = np.random.randint(1950, 2023, N)
lot_area = np.random.randint(2000, 20000, N)
neighborhood = np.random.choice(
    ["Downtown", "Suburb", "Rural", "Uptown", "Riverside"], N,
    p=[0.2, 0.35, 0.15, 0.15, 0.15]
)
house_style = np.random.choice(
    ["1Story", "2Story", "SplitLevel", "Bungalow"], N
)
central_air = np.random.choice(["Y", "N"], N, p=[0.85, 0.15])

neighborhood_premium = {
    "Downtown": 45000, "Suburb": 10000, "Rural": -20000,
    "Uptown": 30000, "Riverside": 25000
}

# Build price from a realistic-ish linear-plus-noise formula
price = (
    20000
    + overall_qual * 18000
    + gr_liv_area * 55
    + garage_cars * 9000
    + total_bsmt_sf * 20
    + full_bath * 7000
    + (year_built - 1950) * 300
    + lot_area * 1.5
    + np.array([neighborhood_premium[n] for n in neighborhood])
    + (central_air == "Y") * 8000
    + np.random.normal(0, 15000, N)
)
price = np.clip(price, 40000, None).round(0)

df = pd.DataFrame({
    "OverallQual": overall_qual,
    "GrLivArea": gr_liv_area,
    "GarageCars": garage_cars,
    "TotalBsmtSF": total_bsmt_sf,
    "FullBath": full_bath,
    "YearBuilt": year_built,
    "LotArea": lot_area,
    "Neighborhood": neighborhood,
    "HouseStyle": house_style,
    "CentralAir": central_air,
    "SalePrice": price,
})

df.to_csv("data/house_prices.csv", index=False)
print(f"Generated {len(df)} rows -> data/house_prices.csv")
print(df["SalePrice"].describe())
