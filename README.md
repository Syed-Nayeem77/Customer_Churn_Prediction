# Telco Churn Prediction Model

This repository contains the machine learning model for predicting customer churn in a telecommunications company.

---

##  Model Structure

The model artifacts are stored in the `models/` directory with the following structure:

```

models/
└── v1/
├── model.pkl         # Serialized trained model (pickle format)
├── features.pkl      # List of feature names used in training
└── metadata.json     # Training information (metrics, version, etc.)

````

---

##  Usage

### Loading the Model

Use the `TelcoChurnModel` class from `load_model.py`:

from models.load_model import TelcoChurnModel

# Initialize the model (defaults to v1)
model = TelcoChurnModel(version="v1")

# Make a prediction
input_data = {
    'feature1': value1,
    'feature2': value2,
    # ... all required features
}
prediction = model.predict(input_data)  # Returns churn probability (0-1)
````

---

### Required Features

The model expects input with the following features (check `features.pkl` for exact list):

* Account length
* Monthly charges
* Total charges
* Contract type
* Payment method
* ... \[add your actual features]

---

##  Requirements

* Python 3.8+

### Dependencies

* scikit-learn
* pandas
* joblib

Install all dependencies with:


pip install -r requirements.txt
```



##  Versioning

Model versions are stored in separate subdirectories (v1, v2, etc.). Each contains:

* The serialized model
* Feature list
* Training metadata

---

##  Training

To retrain the model:


python train.py
```
