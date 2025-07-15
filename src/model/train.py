from sklearn.ensemble import RandomForestClassifier
from .preprocess import preprocess_data
from src.config import settings
import joblib

def train_model():
    df = preprocess_data()
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    
    joblib.dump(model, settings.MODEL_PATH)
    return model
