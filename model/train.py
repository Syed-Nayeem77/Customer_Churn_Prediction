import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import logging

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data():
    """Load and preprocess raw data"""
    try:
        df = pd.read_csv(settings.RAW_DATA_PATH)
        logger.info(f"Loaded raw data from {settings.RAW_DATA_PATH}")
        
        # Convert TotalCharges to numeric, handling errors
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Drop rows with missing values
        df = df.dropna()
        logger.info(f"Data shape after dropping NA: {df.shape}")
        
        # Convert categorical variables to dummy variables
        df = pd.get_dummies(df, drop_first=True)
        
        # Ensure output directory exists
        Path(settings.PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
        df.to_csv(settings.PROCESSED_DATA_PATH, index=False)
        logger.info(f"Processed data saved to {settings.PROCESSED_DATA_PATH}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

def train_model():
    """Train and save model"""
    try:
        # Preprocess data
        df = preprocess_data()
        
        # Prepare features and target
        X = df.drop('Churn_Yes', axis=1)
        y = df['Churn_Yes']
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=settings.TEST_SIZE, 
            random_state=settings.RANDOM_STATE,
            stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=settings.RANDOM_STATE,
            class_weight='balanced'
        )
        logger.info("Training RandomForest model...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        logger.info("\n" + classification_report(y_test, y_pred))
        
        # Save model and features
        Path(settings.MODEL_DIR).mkdir(parents=True, exist_ok=True)
        joblib.dump(model, settings.MODEL_PATH)
        joblib.dump(list(X.columns), settings.FEATURES_PATH)
        logger.info(f"Model saved to {settings.MODEL_PATH}")
        logger.info(f"Feature names saved to {settings.FEATURES_PATH}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()
