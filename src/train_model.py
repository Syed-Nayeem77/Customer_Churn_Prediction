{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a9369265-27b6-4b9c-9d0d-40ad517e92fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "# src/train_model.py\n",
    "import pandas as pd\n",
    "import joblib\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Load data\n",
    "df = pd.read_csv(\"data/customer_churn_data.csv\")\n",
    "\n",
    "# Preprocess\n",
    "df['InternetService'].fillna(df['InternetService'].mode()[0], inplace=True)\n",
    "label_cols = ['Gender', 'ContractType', 'InternetService', 'TechSupport']\n",
    "encoders = {}\n",
    "\n",
    "for col in label_cols:\n",
    "    le = LabelEncoder()\n",
    "    df[col] = le.fit_transform(df[col])\n",
    "    encoders[col] = le\n",
    "    joblib.dump(le, f'model/encoder_{col.lower()}.pkl')\n",
    "\n",
    "df['Churn'] = LabelEncoder().fit_transform(df['Churn'])\n",
    "\n",
    "# Features/labels\n",
    "X = df.drop(['CustomerID', 'Churn'], axis=1)\n",
    "y = df['Churn']\n",
    "\n",
    "# Scaling\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "joblib.dump(scaler, 'model/scaler.pkl')\n",
    "\n",
    "# Train/test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Model\n",
    "model = XGBClassifier()\n",
    "model.fit(X_train, y_train)\n",
    "joblib.dump(model, 'model/churn_model.pkl')"
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
