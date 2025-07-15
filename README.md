# Telco Churn Prediction Model

This repository contains the machine learning model and API for predicting customer churn in a telecommunications company.

---

##  Project Structure

```

Customer\_Churn\_Prediction/
├── api/                      # FastAPI implementation
│   ├── app.py                # FastAPI application
│   └── Dockerfile            # Docker configuration
├── config/                   # Configuration files
│   └── settings.py           # Project settings
├── data/                     # Data files
│   ├── raw/                  # Raw input data
│   └── processed/            # Cleaned/prepared data
├── models/                   # ML model artifacts
│   ├── model.pkl             # Serialized trained model
│   ├── features.pkl          # Feature names used in training
│   ├── metadata.json         # Training details and evaluation metrics
│   └── preprocessor.pkl      # Preprocessing pipeline (e.g., scaler, encoder)
├── monitoring/               # Monitoring and drift analysis
│   └── drift\_report.html     # Evidently AI report
├── notebooks/                # Jupyter notebooks
│   └── EDA.ipynb             # Exploratory Data Analysis
├── src/                      # Source code
│   └── config.py             # Shared configuration logic
├── tests/                    # Test cases for code and API
│   └── test\_api.py           # Unit tests for API endpoints
├── .gitignore                # Git ignore rules
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies

```

##  API Deployment with Docker

To build and run the API container:


# Build the Docker image
docker build -t churn-api .

# Run the container (maps port 8000 on host to 8000 in container)
docker run -p 8000:8000 churn-api
````

Once running, the API will be available at:

```
http://localhost:8000
```

You can send POST requests to the `/predict` endpoint with input data to receive churn predictions.

---

##  Model Usage

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
prediction = model.predict(input_data)  # Returns churn probability (0–1)
```

---

##  Required Features

The model expects input with the following features (check `features.pkl` for exact list):

* Account length
* Monthly charges
* Total charges
* Contract type
* Payment method
* ... \[Add your full list of actual features]

---

##  Requirements

* Python 3.8+

### Install Dependencies

pip install -r requirements.txt
```

Dependencies include:

* `scikit-learn`
* `pandas`
* `joblib`

---

##  Model Versioning

Model versions are stored in subdirectories (`v1`, `v2`, etc.) inside the `models/` folder. Each version contains:

* `model.pkl` — the trained model
* `features.pkl` — the list of features used
* `metadata.json` — training information and evaluation metrics

---

##  Training the Model

To retrain the model with new data:


python train.py
```

This will generate new model artifacts under a versioned directory inside `models/`.
