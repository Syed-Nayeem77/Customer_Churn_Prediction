{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4f8e48f5-23a4-49bb-ba18-516a06f0e747",
   "metadata": {},
   "outputs": [],
   "source": [
    "# main.py\n",
    "from src.predict import predict_churn\n",
    "\n",
    "sample_input = {\n",
    "    'CustomerID': 1001,\n",
    "    'Age': 35,\n",
    "    'Gender': 'Male',\n",
    "    'Tenure': 12,\n",
    "    'MonthlyCharges': 65.5,\n",
    "    'ContractType': 'Month-to-month',\n",
    "    'InternetService': 'DSL',\n",
    "    'TotalCharges': 786.0,\n",
    "    'TechSupport': 'Yes'\n",
    "}\n",
    "\n",
    "print(\"Prediction:\", predict_churn(sample_input))\n"
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
