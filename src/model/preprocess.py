import pandas as pd
from src.config import settings

def preprocess_data():
    df = pd.read_csv(settings.RAW_DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    return pd.get_dummies(df, drop_first=True)
