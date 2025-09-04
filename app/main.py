from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import sys
import glob

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import training and model loading functions
from src.load_model import load_latest_model

app = FastAPI(title="Sales Prediction API")


#Configuration

MODEL_PATH = "models/"
FEATURE_DIR = "models/features"
REQUIRED_COLUMNS = ["priceeach", "quantityordered", "productline","productcode", "customername", "country", "sales"]

numerical_features = ["priceeach", "quantityordered"]
categorical_prefixes = ["productline", "productcode", "customername", "country"]


# Load latest model

try:
    model = load_latest_model(MODEL_PATH)
    # Get the feature names directly from the loaded model's booster
    TRAINING_FEATURES = model.get_booster().feature_names
    print(f"Loaded features from model: {TRAINING_FEATURES[:10]} ...")  # Display first 10 features
except FileNotFoundError:
    model = None
    TRAINING_FEATURES = []
    print("No model found. TRAINING_FEATURES list is empty.")


# Input Schema
class PredictionInput(BaseModel):
    priceeach: float
    quantityordered: int
    productline: str
    productcode: str
    customername: str
    country: str


# Preprocessing Input fields.

def process_data_for_prediction(df: pd.DataFrame, training_features: list):
    """
    Preprocess incoming data for prediction.
    Ensures one-hot encoded features match training features.
    """
    # Normalize column names to lowercase
    df.columns = [col.strip().lower() for col in df.columns]

    # One-hot encode categorical variables
    df_processed = pd.get_dummies(df, columns=categorical_prefixes)

    # Reindex the dataframe to match the training features,
    # filling missing columns with 0.
    df_processed = df_processed.reindex(columns=training_features, fill_value=0)

    # Drop any extra columns not in training
    # This is implicitly handled by reindex, but good for clarity/double-checking
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
        # Pass the features directly from the loaded model
        processed_df = process_data_for_prediction(df, TRAINING_FEATURES)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data preprocessing failed: {str(e)}")

    try:
        prediction = model.predict(processed_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return {"sales_prediction($)": float(prediction[0])}