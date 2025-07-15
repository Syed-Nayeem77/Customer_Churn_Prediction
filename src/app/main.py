from fastapi import FastAPI
from .schemas import CustomerData
from src.model.predict import predict_churn
from src.config import settings
import joblib

app = FastAPI()
model = joblib.load(settings.MODEL_PATH)

@app.post("/predict")
async def predict(data: CustomerData):
    return {"churn_probability": predict_churn(data.dict())}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
