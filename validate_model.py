import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Model
print("Training Random Forest Regressor...")
rf = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2, random_state=42)
rf.fit(X_train, y_train)

# --- VALIDATION CHECKS ---

# Check 1: Overfitting/Underfitting (Train vs Test Gap)
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("\n--- Check 1: Overfitting vs Underfitting Analysis ---")
print(f"Training R2 Score: {r2_train:.4f}")
print(f"Test R2 Score:     {r2_test:.4f}")
print(f"Gap:               {r2_train - r2_test:.4f}")

if r2_train > 0.99 and r2_test < 0.80:
    print(">> DIAGNOSIS: High probability of OVERFITTING (Memorizing training data).")
elif r2_train < 0.60 and r2_test < 0.60:
    print(">> DIAGNOSIS: High probability of UNDERFITTING (Model is too simple).")
else:
    print(">> DIAGNOSIS: Model looks BALANCED. Small gap indicates good generalization.")

# Check 2: Cross-Validation (Stability Check)
print("\n--- Check 2: Cross-Validation (5-Fold) ---")
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
print(f"Cross-Validation R2 Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

if cv_scores.std() > 0.1:
    print(">> WARNING: High variance in CV scores. Model might be unstable on different data subsets.")
else:
    print(">> SUCCESS: Low variance. Model performance is stable.")

# Check 3: Residual Analysis
print("\n--- Check 3: Residual Analysis ---")
residuals = y_test - y_pred_test
mean_residual = np.mean(residuals)
print(f"Mean Residual: {mean_residual:.2f} (Should be close to 0)")

# Save Residual Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_test, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot (Random Pattern = Good, Curve/Shape = Bias)')
plt.savefig('residual_plot.png')
print(">> Residual plot saved to 'residual_plot.png'. Check this image.")
print("   - If points are randomly scattered around 0: GOOD.")
print("   - If points form a curve or funnel: BAD (Model missing patterns).")

# Save results to file for easy reading
with open('validation_report.txt', 'w') as f:
    f.write("--- Model Robustness Validation Report ---\n\n")
    f.write("1. Overfitting Check (Train vs Test):\n")
    f.write(f"   Training R2: {r2_train:.4f}\n")
    f.write(f"   Test R2:     {r2_test:.4f}\n")
    f.write(f"   Gap:         {r2_train - r2_test:.4f}\n\n")
    f.write("2. Stability Check (Cross-Validation):\n")
    f.write(f"   Mean CV R2:  {cv_scores.mean():.4f}\n")
    f.write(f"   Std Dev:     {cv_scores.std():.4f}\n\n")
    f.write("3. Bias Check (Residuals):\n")
    f.write(f"   Mean Residual: {mean_residual:.2f}\n")
