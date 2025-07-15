import pandas as pd
import joblib
from config.settings import settings

class ChurnPredictor:
    def __init__(self):
        self.model = joblib.load(settings.MODEL_PATH)
        self.features = joblib.load(settings.FEATURES_PATH)

    def predict(self, input_data: dict) -> float:
        """Predict churn probability"""
        df = pd.DataFrame([input_data])
        df = pd.get_dummies(df).reindex(columns=self.features, fill_value=0)
        return float(self.model.predict_proba(df)[0][1])
