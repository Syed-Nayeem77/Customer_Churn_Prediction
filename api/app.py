# Customer_Churn_Prediction/api/app.py
from fastapi import FastAPI
from models.load_model import TelcoChurnModel  # Using your existing model class
from pydantic import BaseModel
import os

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="API for predicting customer churn probability"
)

# Load model (with error handling)
try:
    predictor = TelcoChurnModel(version=os.getenv("MODEL_VERSION", "v1"))
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

class CustomerData(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    # Add all other features from your features.pkl
    
    class Config:
        schema_extra = {
            "example": {
                "customerID": "1234-ABCDE",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                # ... other example values
            }
        }

@app.post("/predict")
async def predict(data: CustomerData):
    try:
        prediction = predictor.predict(data.dict())
        return {
            "customerID": data.customerID,
            "churn_probability": round(prediction, 4),
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}

@app.get("/health")
async def health():
    return {
        "status": "OK",
        "model_version": predictor.metadata.get("model_version"),
        "model_type": predictor.metadata.get("algorithm")
    }

@app.get("/model-info")
async def model_info():
    return predictor.metadata
