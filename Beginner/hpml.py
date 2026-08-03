"""
House Price Prediction - End-to-End Pipeline
==============================================

This script demonstrates a complete workflow:
1. Load data (synthetic sample included; swap in your own CSV)
2. Feature engineering
3. Train/test split
4. Train multiple models (Linear Regression, Random Forest, XGBoost)
5. Evaluate & compare
6. Predict on new houses

TO USE YOUR OWN DATA:
Replace `generate_sample_data()` with:
    df = pd.read_csv("your_file.csv")
and make sure your target column is named "price" (or update TARGET_COL below).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

TARGET_COL = "price"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
def generate_sample_data(n=2000):
    """Creates a synthetic but realistic housing dataset for demonstration."""
    rng = np.random.default_rng(RANDOM_STATE)

    sqft = rng.normal(1800, 700, n).clip(400, 6000)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.integers(1, 4, n) + rng.choice([0, 0.5], n)
    year_built = rng.integers(1950, 2023, n)
    lot_size = rng.normal(6000, 2500, n).clip(1000, 20000)
    garage = rng.integers(0, 3, n)
    neighborhood = rng.choice(
        ["Downtown", "Suburb_A", "Suburb_B", "Rural", "Waterfront"],
        n, p=[0.25, 0.25, 0.25, 0.15, 0.10]
    )
    condition = rng.choice(["Poor", "Fair", "Good", "Excellent"], n, p=[0.05, 0.2, 0.5, 0.25])

    # Neighborhood base price multipliers (simulates location being the biggest driver)
    neighborhood_mult = pd.Series(neighborhood).map({
        "Downtown": 1.35, "Waterfront": 1.6, "Suburb_A": 1.0,
        "Suburb_B": 0.9, "Rural": 0.65
    }).values
    condition_mult = pd.Series(condition).map({
        "Poor": 0.75, "Fair": 0.9, "Good": 1.0, "Excellent": 1.2
    }).values

    age = 2024 - year_built
    base_price = (
        sqft * 150
        + bedrooms * 8000
        + bathrooms * 6000
        + lot_size * 3
        + garage * 5000
        - age * 300
    )
    price = base_price * neighborhood_mult * condition_mult
    price += rng.normal(0, 15000, n)  # noise
    price = price.clip(50000, None)

    df = pd.DataFrame({
        "sqft": sqft.round(0),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "lot_size": lot_size.round(0),
        "garage_spaces": garage,
        "neighborhood": neighborhood,
        "condition": condition,
        "price": price.round(0),
    })
    return df


df = generate_sample_data()
print(f"Dataset shape: {df.shape}")
print(df.head())

# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
df["house_age"] = 2024 - df["year_built"]
df["price_per_sqft_proxy"] = df["lot_size"] / df["sqft"]  # example derived feature

numeric_features = ["sqft", "bedrooms", "bathrooms", "lot_size",
                     "garage_spaces", "house_age", "price_per_sqft_proxy"]
categorical_features = ["neighborhood", "condition"]

X = df[numeric_features + categorical_features]
y = np.log1p(df[TARGET_COL])  # log-transform target to reduce skew

# ---------------------------------------------------------------------------
# 3. TRAIN/TEST SPLIT
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 4. PREPROCESSING PIPELINE
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE),
    "XGBoost": XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, random_state=RANDOM_STATE),
}

# ---------------------------------------------------------------------------
# 5. TRAIN + EVALUATE
# ---------------------------------------------------------------------------
results = []
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    preds_log = pipe.predict(X_test)
    preds = np.expm1(preds_log)          # back-transform to real dollars
    actual = np.expm1(y_test)

    mae = mean_absolute_error(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))
    r2 = r2_score(actual, preds)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

results_df = pd.DataFrame(results).sort_values("RMSE")
print("\nModel comparison (on held-out test set):")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. PREDICT ON A NEW HOUSE
# ---------------------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_pipeline = fitted_pipelines[best_model_name]
print(f"\nBest model: {best_model_name}")

new_house = pd.DataFrame([{
    "sqft": 2100,
    "bedrooms": 3,
    "bathrooms": 2.5,
    "lot_size": 7000,
    "garage_spaces": 2,
    "house_age": 2024 - 2015,
    "price_per_sqft_proxy": 7000 / 2100,
    "neighborhood": "Suburb_A",
    "condition": "Good",
}])

pred_log = best_pipeline.predict(new_house)
pred_price = np.expm1(pred_log)[0]
print(f"\nPredicted price for example house: ${pred_price:,.0f}")

# ---------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (for tree-based models)
# ---------------------------------------------------------------------------
if best_model_name in ["Random Forest", "XGBoost"]:
    feature_names = (
        numeric_features
        + list(best_pipeline.named_steps["preprocess"]
               .named_transformers_["cat"]
               .get_feature_names_out(categorical_features))
    )
    importances = best_pipeline.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(10)
    print("\nTop 10 most important features:")
    print(imp_df.to_string(index=False))    