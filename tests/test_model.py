import pytest
from src.model.train import train_model
from src.model.predict import predict_churn

def test_model_training():
    model = train_model()
    assert hasattr(model, 'predict_proba')

def test_prediction():
    test_data = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        # Add required test features
    }
    assert 0 <= predict_churn(test_data) <= 1
