from pathlib import Path

class Settings:
    BASE_DIR = Path(__file__).parent.parent.parent
    RAW_DATA_PATH = BASE_DIR / "data/raw/churn_data.csv"
    MODEL_PATH = BASE_DIR / "models/churn_model.pkl"

settings = Settings()
