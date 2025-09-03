from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import io
import os
import sys

# Add project root (ForecastX-) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train_log_models import train_and_log_model
from src.process_data import feature_engineering
from src.load_model import load_latest_model

app = FastAPI(title="Sales Prediction API")

MODEL_PATH = "models/"

# Load model if exists
try:
    model = load_latest_model(MODEL_PATH)
except FileNotFoundError:
    model = None


# Input Schema
class PredictionInput(BaseModel):
    date: str
    product_id: int
    sales: float


# Endpoint: Predict 
@app.post("/predict")
def predict(input_data: PredictionInput):
    global model
    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet.")

    # Convert input to DataFrame
    df = pd.DataFrame([input_data.dict()])

    # Apply feature engineering
    features = feature_engineering(df)

    # Make prediction
    prediction = model.predict(features)
    return {"prediction": float(prediction[0])}


# Endpoint: File Upload (Retrain) 
@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    global model
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    # Read uploaded CSV
    contents = await file.read()
    data = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Retrain model
    model = train_and_log_model(data)

    # Save updated model
    joblib.dump(model, os.path.join(MODEL_PATH, "best_model_latest.pkl"))

    # Return confirmation + test prediction
    sample = data.sample(1, random_state=42)
    features = feature_engineering(sample.copy())
    pred = model.predict(features)

    return {
        "message": "Model retrained successfully!",
        "sample_input": sample.to_dict(orient="records")[0],
        "sample_prediction": float(pred[0]),
    }
