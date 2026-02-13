# Fraud Detection Project

This repository contains a credit card fraud detection pipeline built using Python, scikit-learn, and feature engineering techniques. The goal is to detect fraudulent transactions in an imbalanced dataset.

---

## 💾 Folder Structure

```
fraud_detection_project/
│
├── models/                 # Trained models and preprocessing objects
│   ├── fraud_detection_pipeline.pkl
│   ├── scaler_amount_standard.pkl
│   ├── scaler_time_standard.pkl
│   ├── amount_bin_edges.pkl
│   └── model_feature_columns.pkl
│
├── notebooks/              # Jupyter notebooks for exploration and feature engineering
│   └── fraud_feature_engineering.ipynb
│
├── data/                   # Raw & processed datasets
│   ├── train.csv
│   └── test.csv
│
├── src/                    # Scripts for training, preprocessing, and predictions
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Load the pipeline and preprocessing objects

```python
import joblib

# Load trained pipeline
pipeline = joblib.load("models/fraud_detection_pipeline.pkl")

# Load scalers
scaler_amount = joblib.load("models/scaler_amount_standard.pkl")
scaler_time = joblib.load("models/scaler_time_standard.pkl")

# Load amount bin edges and feature columns
amount_bin_edges = joblib.load("models/amount_bin_edges.pkl")
feature_columns = joblib.load("models/model_feature_columns.pkl")
```

### 3. Make predictions

```python
# Assume X_new is a new dataset
X_new_scaled = scaler_amount.transform(X_new[['Amount']])
X_new['scaled_amount'] = X_new_scaled
# Add more preprocessing as in your pipeline...

predictions = pipeline.predict(X_new[feature_columns])
```

---

## 📊 Features

- **Scaled features**: `Amount`, `Time`
- **Time-based features**: `hour_of_day`, `time_bin`
- **Amount-based features**: `amount_deviation`, `amount_bin`
- **V-feature statistics**: mean, std
- **Interaction features** between top correlated variables
- **One-hot encoded categorical variables**: `time_bin`, `amount_bin`

---

## 🌈 Model

- Stacking classifier with SMOTE oversampling
- Handles extremely imbalanced dataset (fraud cases ~0.17%)

---

## ⚠️ Notes

- Do **not remove duplicates** in a single column like `Amount` or `Time`
- Always **fit scalers and bins on TRAIN only** to avoid data leakage
- Evaluate models using **precision, rec