{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b9f1547a-894d-4de4-9b97-4d6e0d871bf1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# src/predict.py\n",
    "\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# Load model and artifacts\n",
    "model = joblib.load('model/churn_model.pkl')\n",
    "scaler = joblib.load('model/scaler.pkl')\n",
    "le_gender = joblib.load('model/encoder_gender.pkl')\n",
    "le_contract = joblib.load('model/encoder_contract.pkl')\n",
    "le_internet = joblib.load('model/encoder_internet.pkl')\n",
    "le_tech = joblib.load('model/encoder_tech.pkl')\n",
    "\n",
    "# Example input\n",
    "sample = {\n",
    "    'Age': 35,\n",
    "    'Gender': 'Male',\n",
    "    'Tenure': 12,\n",
    "    'MonthlyCharges': 70.5,\n",
    "    'ContractType': 'One year',\n",
    "    'InternetService': 'Fiber optic',\n",
    "    'TotalCharges': 845.0,\n",
    "    'TechSupport': 'Yes'\n",
    "}\n",
    "\n",
    "# Encode categorical features\n",
    "sample_encoded = [\n",
    "    sample['Age'],\n",
    "    le_gender.transform([sample['Gender']])[0],\n",
    "    sample['Tenure'],\n",
    "    sample['MonthlyCharges'],\n",
    "    le_contract.transform([sample['ContractType']])[0],\n",
    "    le_internet.transform([sample['InternetService']])[0],\n",
    "    sample['TotalCharges'],\n",
    "    le_tech.transform([sample['TechSupport']])[0],\n",
    "]\n",
    "\n",
    "# Scale numeric values\n",
    "sample_scaled = scaler.transform([sample_encoded])\n",
    "\n",
    "# Predict\n",
    "prediction = model.predict(sample_scaled)[0]\n",
    "print(\"Prediction (1=Churn, 0=No Churn):\", prediction)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
