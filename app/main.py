# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import io
import os
import sys
import glob
import chardet

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import training and model loading functions
from src.train_log_models import train_and_log_model
from src.load_model import load_latest_model

app = FastAPI(title="Sales Prediction API")

# Configuration
MODEL_PATH = "models/"
FEATURE_DIR = "models/features"
REQUIRED_COLUMNS = ["priceeach", "quantityordered", "productline", "productcode", "customername", "country", "sales"]

numerical_features = ["priceeach", "quantityordered"]
categorical_prefixes = ["productline", "productcode", "customername", "country"]

# Load latest model
try:
    model = load_latest_model(MODEL_PATH)
except FileNotFoundError:
    model = None

# Load features
feature_files = glob.glob(os.path.join(FEATURE_DIR, "*.txt"))
if not feature_files:
    raise FileNotFoundError(f"No feature file found in {FEATURE_DIR}")

latest_feature_file = max(feature_files, key=os.path.getmtime)
with open(latest_feature_file, "r") as f:
    all_features = f.read().splitlines()

# Expand categorical features based on one-hot encoding
expanded_categorical_features = []
for prefix in categorical_prefixes:
    expanded_categorical_features.extend([f for f in all_features if f.startswith(prefix + "_")])

TRAINING_FEATURES = numerical_features + expanded_categorical_features
TARGET = "sales"

print(f"Loaded features: {TRAINING_FEATURES[:10]} ...")  # Display first 10 features

# Input Schema
class PredictionInput(BaseModel):
    priceeach: float
    quantityordered: int
    productline: str
    productcode: str
    customername: str
    country: str

# Preprocessing
def process_data_for_prediction(df: pd.DataFrame, training_features: list):
    """
    Preprocess incoming data for prediction.
    Ensures one-hot encoded features match training features.
    """
    # Normalize column names to lowercase
    df.columns = [col.strip().lower() for col in df.columns]

    df_processed = pd.get_dummies(df, columns=categorical_prefixes)

    # Add missing columns from training features
    for col in training_features:
        if col not in df_processed.columns:
            df_processed[col] = 0

    # Reorder columns to match training features
    df_processed = df_processed[training_features]

    return df_processed
    

# Prediction Endpoint
@app.post("/predict")
def predict(input_data: PredictionInput):
    global model
    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet.")

    df = pd.DataFrame([input_data.dict()])

    try:
        processed_df = process_data_for_prediction(df, TRAINING_FEATURES)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data preprocessing failed: {str(e)}")

    prediction = model.predict(processed_df)
    return {"prediction": float(prediction[0])}

