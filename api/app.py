from fastapi import FastAPI
from model.predict import ChurnPredictor
from pydantic import BaseModel

app = FastAPI()
predictor = ChurnPredictor()

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    # Add all other features from dataset

@app.post("/predict")
async def predict(data: CustomerData):
    return {"churn_probability": predictor.predict(data.dict())}

@app.get("/health")
async def health():
    return {"status": "OK"}
