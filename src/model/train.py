import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import json
from pathlib import Path
from datetime import datetime
from src.config import settings
import logging
from .preprocess import preprocess_data
from .evaluate import evaluate_model

logger = logging.getLogger(__name__)

def save_model_artifacts(model, feature_names, metrics):
    """Save model and metadata"""
    try:
        Path("models").mkdir(exist_ok=True)
        
        # Save artifacts
        joblib.dump(model, settings.MODEL_PATH)
        joblib.dump(feature_names, settings.FEATURES_PATH)
        
        metadata = {
            "version": "1.0.0",
            "training_date": datetime.now().isoformat(),
            "metrics": metrics,
            "features": str(settings.FEATURES_PATH),
            "model_type": "RandomForestClassifier"
        }
        
        with open(settings.METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Model artifacts saved")
        
    except Exception as e:
        logger.error(f"Failed to save artifacts: {str(e)}")
        raise

def train_model():
    """Orchestrate training pipeline"""
    try:
        # Load and preprocess
        df = pd.read_csv(settings.RAW_DATA_PATH)
        df = preprocess_data(df)
        
        # Prepare data
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save
        save_model_artifacts(
            model=model,
            feature_names=list(X.columns),
            metrics=metrics
        )
        
        return model
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()
