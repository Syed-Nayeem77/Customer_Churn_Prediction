from pathlib import Path

class Settings:
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    MODEL_DIR = BASE_DIR / "models"
    
    # File paths
    MODEL_PATH = MODEL_DIR / "churn_model_v1.pkl"
    FEATURES_PATH = MODEL_DIR / "features.pkl"
    METADATA_PATH = MODEL_DIR / "metadata.json"
    RAW_DATA_PATH = BASE_DIR / "data/raw/churn_data.csv"
    
    # Training
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

settings = Settings()
