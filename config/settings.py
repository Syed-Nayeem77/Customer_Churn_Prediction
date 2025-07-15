from pathlib import Path

class Settings:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    RAW_DATA_PATH = BASE_DIR / "data/raw/churn_data.csv"
    PROCESSED_DATA_DIR = BASE_DIR / "data/processed"
    PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_data.csv"
    MODEL_DIR = BASE_DIR / "models"
    MODEL_PATH = MODEL_DIR / "churn_model.pkl"
    
    # Training
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

settings = Settings()
