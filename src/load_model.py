import os
import glob
import joblib

def load_latest_model(model_dir: str = "models"):
    """
    Load the most recently saved model from the given directory.
    """
    pattern = os.path.join(model_dir, "best_model_*.pkl")
    model_files = glob.glob(pattern)

    if not model_files:
        raise FileNotFoundError("No latest model pickle files found in 'models/' directory.")

    # Sort files by modification time
    model_files.sort(key=os.path.getmtime)
    latest_model_path = model_files[-1]
    print(f"Latest model: {latest_model_path}")

    # Load and return model object
    return joblib.load(latest_model_path)
