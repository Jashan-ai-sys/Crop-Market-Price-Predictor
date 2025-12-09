import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Load Data
print("Loading full dataset...")
df = pd.read_csv('final_forecasting_dataset.csv')

# 2. Feature Engineering (Leakage Fix)
print("Applying feature engineering...")
df['Modal_Price_Corrected_Roll7'] = df['Modal_Price'].shift(1).rolling(window=7).mean()
df['Modal_Price_Corrected_Roll30'] = df['Modal_Price'].shift(1).rolling(window=30).mean()

# 3. Drop NaNs
df_clean = df.dropna()

# 4. Define Features
features = [
    'Season_Num', 'Rainfall_mm', 'Temp_C', 
    'Rainfall_Lag7', 'Temp_Lag7', 'Festival_Flag',
    'Modal_Price_Lag7', 'Modal_Price_Lag30', 
    'Modal_Price_Corrected_Roll7', 'Modal_Price_Corrected_Roll30',
    'Price_Momentum', 'Fourier_sin', 'Fourier_cos'
]

X = df_clean[features]
y = df_clean['Modal_Price']

# 5. Train Model on Full Data
print("Training model on complete dataset...")
rf = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2, random_state=42)
rf.fit(X, y)

# 6. Save Model
model_filename = 'crop_price_model.joblib'
joblib.dump(rf, model_filename)
print(f"Model saved to {model_filename}")

# 7. Save Feature Names (to ensure correct order during inference)
joblib.dump(features, 'model_features.joblib')
print("Feature list saved to model_features.joblib")
