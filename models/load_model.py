import joblib
import json
from pathlib import Path

class TelcoChurnModel:
    def __init__(self, version="v1"):
        self.model_dir = Path(f"models/{version}")
        try:
            self.model = joblib.load(self.model_dir / "model.pkl")
            self.features = joblib.load(self.model_dir / "features.pkl")
            with open(self.model_dir / "metadata.json") as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise ValueError(f"Model loading failed: {str(e)}")

    def predict(self, input_data):
        """input_data: dict of feature values"""
        import pandas as pd
        df = pd.DataFrame([input_data])
        return float(self.model.predict_proba(df)[0][1])
