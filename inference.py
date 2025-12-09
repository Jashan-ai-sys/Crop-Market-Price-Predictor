import joblib
import pandas as pd
import numpy as np

# 1. Load Model and Features
print("Loading model...")
model = joblib.load('crop_price_model.joblib')
features = joblib.load('model_features.joblib')

# 2. Sample Input Data (Simulating a new row for prediction)
# NOTE: In a real scenario, you need to calculate Rolling averages based on history.
# Here we take a known row from the dataset for demonstration.
sample_data = {
    'Season_Num': [1], 
    'Rainfall_mm': [10.5], 
    'Temp_C': [25.0], 
    'Rainfall_Lag7': [5.0], 
    'Temp_Lag7': [24.0], 
    'Festival_Flag': [0],
    'Modal_Price_Lag7': [1500], 
    'Modal_Price_Lag30': [1450], 
    'Modal_Price_Corrected_Roll7': [1480],  # Calculated from history
    'Modal_Price_Corrected_Roll30': [1460], # Calculated from history
    'Price_Momentum': [0.02], 
    'Fourier_sin': [0.5], 
    'Fourier_cos': [0.866]
}

input_df = pd.DataFrame(sample_data)

# Ensure correct column order
input_df = input_df[features]

# 3. Predict
print("\n--- Predicting for Sample Input ---")
print(input_df.to_string(index=False))
prediction = model.predict(input_df)

print(f"\n>> Predicted Crop Price: {prediction[0]:.2f}")
