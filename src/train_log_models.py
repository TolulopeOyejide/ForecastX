import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pickle
import glob
import time
from datetime import datetime
import warnings
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso

warnings.filterwarnings("ignore")


def train_and_log_model(data: pd.DataFrame = None, data_dir: str = "data/cleaned/", model_dir: str = "models/", feature_dir: str = "models/features", experiment_name: str = "ForecastX_Regression",):
    """
    Train multiple regression models with GridSearchCV and MLflow tracking.
    Saves the best model to pickle file.

    Args:
        data (pd.DataFrame, optional): Input data. If None, loads latest CSV from `data_dir`.
        data_dir (str): Directory containing cleaned CSV files.
        model_dir (str): Directory to save best model pickle.
        experiment_name (str): MLflow experiment name.

    Returns:
        str: Path to the saved best model.
        str: Name of the best model.
        float: Best model RMSE.
    """

    os.makedirs(model_dir, exist_ok=True)

    
    # Load data
    if data is None:
        # Look specifically for processed files (cleaned_data_*.csv)
        csv_files = glob.glob(os.path.join(data_dir, "cleaned_data_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No processed CSV files (cleaned_data_*.csv) found in {data_dir}")
        latest_file = max(csv_files, key=os.path.getmtime)
        df = pd.read_csv(latest_file)
        print(f"Latest processed file loaded: {latest_file}")
        print(f"Data Info: {df.info()}")
    else:
        df = data.copy()
        print("Data loaded from API upload")


    #Load Features
    # Get all features.txt files in the feature_dir (including subfolders if needed)
    feature_files = glob.glob(os.path.join(feature_dir, "*.txt"))
    if not feature_files:
        raise FileNotFoundError(f"No feature file found in {feature_dir}")
    latest_feature_file = max(feature_files, key=os.path.getmtime)
    with open(latest_feature_file, "r") as f:
        all_features = f.read().splitlines()
    print(f"Features loaded from {latest_feature_file}: {all_features[:10]} ...")
  

    
    numerical_features = ["priceeach", "quantityordered"]
    categorical_feature_prefix = ["productline", "productcode", "customername", "country"]


    expanded_categorical_features = []
    for prefix in categorical_feature_prefix:
        expanded_categorical_features.extend([f for f in all_features if f.startswith(prefix + "_")])

    features_for_model = numerical_features + expanded_categorical_features
    target = "sales"

   
    valid_features = [col for col in features_for_model if col in df.columns]
    missing_features = [col for col in features_for_model if col not in df.columns]
    if missing_features:
        print(f"Warning: Missing features in DataFrame: {missing_features}")



    # Drop rows with NaNs in valid features + target
    df = df.dropna(subset=valid_features + [target])

    X = df[valid_features]
    y = df[target]
    print(X)
    print(y)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)


    # Models dictionary
    models = {
    "LinearRegression": ( Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())]),{"regressor__fit_intercept": [True, False]},),
    "PolynomialRegression": (Pipeline([("poly", PolynomialFeatures(degree=2)), ("scaler", StandardScaler()), ("regressor", LinearRegression())]),{"poly__degree": [2, 3]},),
    "Ridge": (Pipeline([("scaler", StandardScaler()), ("regressor", Ridge())]),{"regressor__alpha": [0.01, 0.1, 1.0, 10.0]},),
    "Lasso": ( Pipeline([("scaler", StandardScaler()), ("regressor", Lasso())]),{"regressor__alpha": [0.001, 0.01, 0.1, 1.0]},),
    "RandomForest": (RandomForestRegressor(random_state=42),{"n_estimators": [100, 200], "max_depth": [5, 10, None]},),
    "XGBoost": (XGBRegressor(random_state=42, eval_metric="rmse"),{"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.01, 0.1]},),
    "KNN": (Pipeline([("scaler", StandardScaler()), ("regressor", KNeighborsRegressor())]),{"regressor__n_neighbors": [3, 5, 7]},),
    "SVM": (Pipeline([("scaler", StandardScaler()), ("regressor", SVR())]),{"regressor__C": [0.1, 1, 10], "regressor__kernel": ["linear", "rbf"]},),
    "NeuralNetwork": (Pipeline([("scaler", StandardScaler()), ("regressor", MLPRegressor(max_iter=1000))]),{"regressor__hidden_layer_sizes": [(50,), (100,)],"regressor__activation": ["relu", "tanh"],
            "regressor__alpha": [0.0001, 0.001],}),}


    #Training Loop
    best_model, best_name, best_score = None, None, float("inf")

    mlflow.set_experiment(experiment_name)

    for name, (model, params) in models.items():
        if mlflow.active_run():
            mlflow.end_run()

        print(f"Training {name}...")
        start_time = time.time()

        grid = GridSearchCV(
            model, params, cv=5, scoring="neg_mean_squared_error", n_jobs=-1
        )

        with mlflow.start_run(run_name=name):
            grid.fit(X_train, y_train)
            elapsed_time = time.time() - start_time

            # Predictions
            y_pred = grid.best_estimator_.predict(X_test)

            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # Log
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            mlflow.sklearn.log_model(grid.best_estimator_, name)

            print(
                f"{name} -> RMSE: {rmse:.4f}, R2: {r2:.4f}, Training Time: {elapsed_time:.2f} sec")

            if rmse < best_score:
                best_score = rmse
                best_model = grid.best_estimator_
                best_name = name


    # Save best model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(model_dir, f"best_model_{timestamp}.pkl")

    with open(best_model_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"\nBest Model: {best_name} with RMSE: {best_score:.4f}")
    print(f"Saved to {best_model_path}")

    return best_model_path, best_name, best_score


if __name__ == "__main__":
    train_and_log_model()