# Crop Market Price Predictor

## 🌾 Project Overview
This project enables accurate forecasting of crop market prices using historical data, weather patterns, and seasonality indicators. It features a robust machine learning pipeline capable of capturing complex temporal dependencies, achieving a high precision in price prediction.

## 🚀 Key Features
*   **High Accuracy:** The final Random Forest model achieves an **R² score of ~0.977** (RMSE: ~213.56).
*   **Leakage-Free Feature Engineering:** rigorously implemented `shift()` based rolling averages to prevent data leakage from future timestamps.
*   **Advanced Seasonality:** Incorporates Fourier terms (`sin`/`cos`) and explicit festival flags (e.g., Diwali, Holi) to capture cyclical market trends.
*   **Comprehensive Analysis:** Includes detailed Exploratory Data Analysis (EDA) on price momentum, weather measurements, and lag features.

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| **`train_model.ipynb`** | **Start Here.** The core notebook that loads the dataset, performs the final leakage-free rolling feature calculation, trains the Random Forest Regressor, and evaluates performance. |
| **`eda.ipynb`** | Detailed Exploratory Data Analysis. Visualizes trends, seasonality, and correlations. *Note: Some intermediate files generated here are precursors to the final dataset.* |
| **`final_forecasting_dataset.csv`** | The complete, clean dataset used for modeling. Contains all base features (Weather, Seasonality, Prices). |
| `improve_model.ipynb` | Experiments with Gradient Boosting Regressors to further optimize performance. |
| `tune_model.ipynb` | Hyperparameter tuning using GridSearchCV to find the optimal model configuration. |
| `farmer_data.csv` | The raw initial dataset. |

## 🛠️ Technology Stack
*   **Python**
*   **Pandas & NumPy** (Data Manipulation & Feature Engineering)
*   **Scikit-Learn** (Random Forest, Gradient Boosting, Metrics)
*   **Matplotlib & Seaborn** (Visualization)

## ⚙️ How to Run
1.  **Clone the repository** (if applicable).
2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  **Train the Model:**
    Open and run all cells in `train_model.ipynb`. This will:
    *   Load `final_forecasting_dataset.csv`.
    *   Recalculate rolling features to ensure no data leakage.
    *   Train the Random Forest Regressor.
    *   Output the Model Performance metrics (MAE, MSE, RMSE, R²).

## 📊 Model Performance
*   **Model:** Random Forest Regressor (`n_estimators=200`)
*   **R² Score:** 0.9771
*   **Mean Absolute Error (MAE):** 101.15
*   **Root Mean Squared Error (RMSE):** 213.56