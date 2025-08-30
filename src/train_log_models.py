import argparse
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


from sklearn.linear_model import LinearRegression #Linear Regression Model
from sklearn.ensemble import RandomForestRegressor #Random Forest Regression Model
from xgboost import XGBRegressor #XGBoost Regression Model
from sklearn.neighbors import KNeighborsRegressor #KNN Regression Model
from sklearn.svm import SVR #Suport Vector Machines
from sklearn.neural_network import MLPRegressor #Neural Network


warnings.filterwarnings("ignore")


#MODEL PATH
MODEL_PATH = "models/"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

#LOAD DATA
DATA_DIR = "data/cleaned/"

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

# Pick the most recently modified file
latest_file = max(csv_files, key=os.path.getmtime)

# Load dataset
df = pd.read_csv(latest_file)
print(f"Latest file loaded: {latest_file}")
print(f"Dataset Columns: {df.columns}")


#SPLIT DATA
features = ['price','promo','marketing_spend','sales_roll_mean_7','sales_roll_std_7']
target = 'sales'

df = df.dropna(subset=features+[target])

X = df[features]
y = df[target] #sales

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# DEFINE THE MODELS
models = {
    "LinearRegression": ( Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())]), {"regressor__fit_intercept": [True, False]}),

    "PolynomialRegression": (Pipeline([("poly", PolynomialFeatures(degree=2)),("scaler", StandardScaler()),("regressor", LinearRegression())]),{"poly__degree": [2, 3]}),

    "RandomForest": (RandomForestRegressor(random_state=42), {"n_estimators": [100, 200], "max_depth": [5, 10, None]}),

    "XGBoost": (XGBRegressor(random_state=42, eval_metric="rmse"),{"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.01, 0.1]}),

    "KNN": ( Pipeline([("scaler", StandardScaler()), ("regressor", KNeighborsRegressor())]), {"regressor__n_neighbors": [3, 5, 7]}),

    "SVM": ( Pipeline([("scaler", StandardScaler()), ("regressor", SVR())]),{"regressor__C": [0.1, 1, 10], "regressor__kernel": ["linear", "rbf"]} ),

    "NeuralNetwork": ( Pipeline([("scaler", StandardScaler()), ("regressor", MLPRegressor(max_iter=1000))]), {"regressor__hidden_layer_sizes": [(50,), (100,)],"regressor__activation": ["relu", "tanh"],
         "regressor__alpha": [0.0001, 0.001]}),
         }


#TRAINING LOOP WITH GRID SEARCH AND MLFLOW TRACKING
best_model = None
best_name = None
best_score = float("inf")

mlflow.set_experiment("ForecastX_Regression")

for name, (model, params) in models.items():
    if mlflow.active_run():
        mlflow.end_run()
        
    print(f"Training {name}...")

    start_time = time.time()  # start timer
    
    grid = GridSearchCV(model, params, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    
    with mlflow.start_run(run_name=name):
        grid.fit(X_train, y_train)

        elapsed_time = time.time() - start_time  # compute elapsed time
    
        # Predictions
        y_pred = grid.best_estimator_.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Log params and metrics
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(grid.best_estimator_, name)
        
        print(f"{name} -> RMSE: {rmse:.4f}, R2: {r2:.4f}, Training Time: {elapsed_time:.2f} seconds")
        
        # Track best model
        if rmse < best_score:
            best_score = rmse
            best_model = grid.best_estimator_
            best_name = name

# -------------------------------
# AFTER the loop: Save and Print best model
# -------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_PATH = f"models/best_model_{timestamp}.pkl"

with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)

print(f"\n✅ Best Model: {best_name} with RMSE: {best_score:.4f}")
print(f"Saved to {MODEL_PATH}")
