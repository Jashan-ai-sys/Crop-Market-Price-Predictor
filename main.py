from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import joblib
import pandas as pd

app = FastAPI(
    title="Crop Price Prediction API",
    description="ML API for crop price prediction",
    version="1.0"
)

# Load artifacts ONCE
try:
    model = joblib.load("crop_price_model.joblib")
    features = joblib.load("model_features.joblib")
except Exception as e:
    raise RuntimeError(f"Startup failed: {e}")

class PredictRequest(BaseModel):
    data: Dict[str, float]

@app.get("/")
def health():
    return {"status": "Crop Price Model is running"}

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([request.data])

        # Ensure correct feature order
        df = df[features]

        # Predict
        price = model.predict(df)[0]

        return {
            "predicted_price": round(float(price), 2)
        }

    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Missing feature: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
