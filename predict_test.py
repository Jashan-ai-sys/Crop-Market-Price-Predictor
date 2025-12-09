import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv('final_forecasting_dataset.csv')

# 2. Feature Engineering (Leakage Fix)
print("Applying feature engineering (Rolling Averages with Shift)...")
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

# 5. Split Data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Model
print("Training Random Forest Regressor...")
rf = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2, random_state=42)
rf.fit(X_train, y_train)

# 7. Predict
print("Generating predictions on test set...")
y_pred = rf.predict(X_test)

# 8. Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

with open('results.txt', 'w') as f:
    f.write(f"--- Model Performance Check ---\n")
    f.write(f"R2 Score: {r2:.4f}\n")
    f.write(f"MAE: {mae:.2f}\n")

    # 9. Sample Predictions
    f.write(f"\n--- Sample Predictions (Actual vs Predicted) ---\n")
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    # Show last 10 predictions from the test set
    f.write(results.head(10).round(2).to_string())
    f.write("\n")

print("Verification complete. Results written to results.txt")
