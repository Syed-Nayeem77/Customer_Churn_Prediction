import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data():
    """Load and preprocess raw data"""
    df = pd.read_csv(settings.RAW_DATA_PATH)
    
    # Example preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    
    Path(settings.PROCESSED_DATA_DIR).mkdir(exist_ok=True)
    df.to_csv(settings.PROCESSED_DATA_PATH, index=False)
    return df

def train_model():
    """Train and save model"""
    df = preprocess_data()
    X = df.drop('Churn_Yes', axis=1)
    y = df['Churn_Yes']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=settings.RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    Path(settings.MODEL_DIR).mkdir(exist_ok=True)
    joblib.dump(model, settings.MODEL_PATH)
    logger.info(f"Model saved to {settings.MODEL_PATH}")
    
    return model

if __name__ == "__main__":
    train_model()
