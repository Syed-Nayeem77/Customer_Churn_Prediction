import pandas as pd
from src.config import settings
import joblib

model = joblib.load(settings.MODEL_PATH)

def predict_churn(input_data: dict) -> float:
    df = pd.DataFrame([input_data])
    return float(model.predict_proba(df)[0][1])
