from pathlib import Path

class Settings:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_PATH = DATA_DIR / "raw/churn_data.csv"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_data.csv"
    
    # Model
    MODEL_DIR = BASE_DIR / "models"
    MODEL_PATH = MODEL_DIR / "churn_model_v1.pkl"
    FEATURES_PATH = MODEL_DIR / "features.pkl"
    
    # Training
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

settings = Settings()
