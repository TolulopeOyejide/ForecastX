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

warnings.filterwarnings("ignore")


def train_and_log_model( data: pd.DataFrame = None,
                         data_dir: str = "data/cleaned/",model_dir: str = "models/",
                         experiment_name: str = "ForecastX_Regression",):
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
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        latest_file = max(csv_files, key=os.path.getmtime)
        df = pd.read_csv(latest_file)
        print(f"Latest file loaded: {latest_file}")
    else:
        df = data.copy()
        print("Data loaded from API upload")

    print(f"Dataset Columns: {df.columns}")

    df.head()

    #Sorting Categorical data
    productline = [
    'productline_Motorcycles',
    'productline_Planes',
    'productline_Ships',
    'productline_Trains',
    'productline_Trucks and Buses',
    'productline_Vintage Cars']


    productcode = [
    'productcode_S10_1949',
    'productcode_S10_2016',
    'productcode_S10_4698',
    'productcode_S10_4757',
    'productcode_S10_4962',
    'productcode_S12_1099',
    'productcode_S12_1108',
    'productcode_S12_1666',
    'productcode_S12_2823',
    'productcode_S12_3148',
    'productcode_S12_3380',
    'productcode_S12_3891',
    'productcode_S12_3990',
    'productcode_S12_4473',
    'productcode_S12_4675',
    'productcode_S18_1097',
    'productcode_S18_1129',
    'productcode_S18_1342',
    'productcode_S18_1367',
    'productcode_S18_1589',
    'productcode_S18_1662',
    'productcode_S18_1749',
    'productcode_S18_1889',
    'productcode_S18_1984',
    'productcode_S18_2238',
    'productcode_S18_2248',
    'productcode_S18_2319',
    'productcode_S18_2325',
    'productcode_S18_2432',
    'productcode_S18_2581',
    'productcode_S18_2625',
    'productcode_S18_2795',
    'productcode_S18_2870',
    'productcode_S18_2949',
    'productcode_S18_2957',
    'productcode_S18_3029',
    'productcode_S18_3136',
    'productcode_S18_3140',
    'productcode_S18_3232',
    'productcode_S18_3259',
    'productcode_S18_3278',
    'productcode_S18_3320',
    'productcode_S18_3482',
    'productcode_S18_3685',
    'productcode_S18_3782',
    'productcode_S18_3856',
    'productcode_S18_4027',
    'productcode_S18_4409',
    'productcode_S18_4522',
    'productcode_S18_4600',
    'productcode_S18_4668',
    'productcode_S18_4721',
    'productcode_S18_4933',
    'productcode_S24_1046',
    'productcode_S24_1444',
    'productcode_S24_1578',
    'productcode_S24_1628',
    'productcode_S24_1785',
    'productcode_S24_1937',
    'productcode_S24_2000',
    'productcode_S24_2011',
    'productcode_S24_2022',
    'productcode_S24_2300',
    'productcode_S24_2360',
    'productcode_S24_2766',
    'productcode_S24_2840',
    'productcode_S24_2841',
    'productcode_S24_2887',
    'productcode_S24_2972',
    'productcode_S24_3151',
    'productcode_S24_3191',
    'productcode_S24_3371',
    'productcode_S24_3420',
    'productcode_S24_3432',
    'productcode_S24_3816',
    'productcode_S24_3856',
    'productcode_S24_3949',
    'productcode_S24_3969',
    'productcode_S24_4048',
    'productcode_S24_4258',
    'productcode_S24_4278',
    'productcode_S24_4620',
    'productcode_S32_1268',
    'productcode_S32_1374',
    'productcode_S32_2206',
    'productcode_S32_2509',
    'productcode_S32_3207',
    'productcode_S32_3522',
    'productcode_S32_4289',
    'productcode_S32_4485',
    'productcode_S50_1341',
    'productcode_S50_1392',
    'productcode_S50_1514',
    'productcode_S50_4713',
    'productcode_S700_1138',
    'productcode_S700_1691',
    'productcode_S700_1938',
    'productcode_S700_2047',
    'productcode_S700_2466',
    'productcode_S700_2610',
    'productcode_S700_2824',
    'productcode_S700_2834',
    'productcode_S700_3167',
    'productcode_S700_3505',
    'productcode_S700_3962',
    'productcode_S700_4002',
    'productcode_S72_1253',
    'productcode_S72_3212']


    customername = [
    'customername_Alpha Cognac',
    'customername_Amica Models & Co.',
    "customername_Anna's Decorations, Ltd",
    'customername_Atelier graphique',
    'customername_Australian Collectables, Ltd',
    'customername_Australian Collectors, Co.',
    'customername_Australian Gift Network, Co',
    'customername_Auto Assoc. & Cie.',
    'customername_Auto Canal Petit',
    'customername_Auto-Moto Classics Inc.',
    'customername_Baane Mini Imports',
    'customername_Bavarian Collectables Imports, Co.',
    'customername_Blauer See Auto, Co.',
    'customername_Boards & Toys Co.',
    'customername_CAF Imports',
    'customername_Cambridge Collectables Co.',
    'customername_Canadian Gift Exchange Network',
    'customername_Classic Gift Ideas, Inc',
    'customername_Classic Legends Inc.',
    'customername_Clover Collections, Co.',
    'customername_Collectable Mini Designs Co.',
    'customername_Collectables For Less Inc.',
    'customername_Corporate Gift Ideas Co.',
    'customername_Corrida Auto Replicas, Ltd',
    'customername_Cruz & Sons Co.',
    'customername_Daedalus Designs Imports',
    'customername_Danish Wholesale Imports',
    'customername_Diecast Classics Inc.',
    'customername_Diecast Collectables',
    'customername_Double Decker Gift Stores, Ltd',
    'customername_Dragon Souveniers, Ltd.',
    'customername_Enaco Distributors',
    'customername_Euro Shopping Channel',
    'customername_FunGiftIdeas.com',
    'customername_Gift Depot Inc.',
    'customername_Gift Ideas Corp.',
    'customername_Gifts4AllAges.com',
    'customername_Handji Gifts& Co',
    'customername_Heintze Collectables',
    'customername_Herkku Gifts',
    'customername_Iberia Gift Imports, Corp.',
    "customername_L'ordine Souveniers",
    "customername_La Corne D'abondance, Co.",
    'customername_La Rochelle Gifts',
    'customername_Land of Toys Inc.',
    'customername_Lyon Souveniers',
    'customername_Marseille Mini Autos',
    "customername_Marta's Replicas Co.",
    "customername_Men 'R' US Retailers, Ltd.",
    'customername_Microscale Inc.',
    'customername_Mini Auto Werke',
    'customername_Mini Caravy',
    'customername_Mini Classics',
    'customername_Mini Creations Ltd.',
    'customername_Mini Gifts Distributors Ltd.',
    'customername_Mini Wheels Co.',
    'customername_Motor Mint Distributors Inc.',
    'customername_Muscle Machine Inc',
    'customername_Norway Gifts By Mail, Co.',
    'customername_Online Diecast Creations Co.',
    'customername_Online Mini Collectables',
    'customername_Osaka Souveniers Co.',
    'customername_Oulu Toy Supplies, Inc.',
    'customername_Petit Auto',
    'customername_Quebec Home Shopping Network',
    'customername_Reims Collectables',
    'customername_Rovelli Gifts',
    'customername_Royal Canadian Collectables, Ltd.',
    'customername_Royale Belge',
    'customername_Salzburg Collectables',
    'customername_Saveley & Henriot, Co.',
    'customername_Scandinavian Gift Ideas',
    'customername_Signal Collectibles Ltd.',
    'customername_Signal Gift Stores',
    'customername_Souveniers And Things Co.',
    'customername_Stylish Desk Decors, Co.',
    'customername_Suominen Souveniers',
    'customername_Super Scale Inc.',
    'customername_Technics Stores Inc.',
    'customername_Tekni Collectables Inc.',
    'customername_The Sharp Gifts Warehouse',
    'customername_Tokyo Collectables, Ltd',
    'customername_Toms Spezialitten, Ltd',
    'customername_Toys of Finland, Co.',
    'customername_Toys4GrownUps.com',
    'customername_UK Collectables, Ltd.',
    'customername_Vida Sport, Ltd',
    'customername_Vitachrome Inc.',
    'customername_Volvo Model Replicas, Co',
    'customername_West Coast Collectables Co.',
    'customername_giftsbymail.co.uk']

    country = [
    'country_Austria',
    'country_Belgium',
    'country_Canada',
    'country_Denmark',
    'country_Finland',
    'country_France',
    'country_Germany',
    'country_Ireland',
    'country_Italy',
    'country_Japan',
    'country_Norway',
    'country_Philippines',
    'country_Singapore',
    'country_Spain',
    'country_Sweden',
    'country_Switzerland',
    'country_UK',
    'country_USA']

    # Combine all one-hot encoded lists into a single list
    one_hot_features = productline + productcode + customername + country

    # Define your other numerical features
    numerical_features = ["priceeach", "quantityordered"]

    # Combine numerical and one-hot encoded features to create the final list
    features = numerical_features + one_hot_features


    target = "sales"

    df = df.dropna(subset=features + [target])

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

    
    # Models dictionary
    models = {"LinearRegression": (Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())]),{"regressor__fit_intercept": [True, False]},),
        "PolynomialRegression":(Pipeline([("poly", PolynomialFeatures(degree=2)),("scaler", StandardScaler()),("regressor", LinearRegression()),]),{"poly__degree": [2, 3]},),
        "RandomForest": (RandomForestRegressor(random_state=42),{"n_estimators": [100, 200], "max_depth": [5, 10, None]},),
        "XGBoost": (XGBRegressor(random_state=42, eval_metric="rmse"),{"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.01, 0.1]},),
        "KNN": (Pipeline([("scaler", StandardScaler()), ("regressor", KNeighborsRegressor())]),{"regressor__n_neighbors": [3, 5, 7]},),
        "SVM": (Pipeline([("scaler", StandardScaler()), ("regressor", SVR())]),{"regressor__C": [0.1, 1, 10], "regressor__kernel": ["linear", "rbf"]},),
        "NeuralNetwork": (Pipeline([("scaler", StandardScaler()), ("regressor", MLPRegressor(max_iter=1000))]),{
                "regressor__hidden_layer_sizes": [(50,), (100,)],
                "regressor__activation": ["relu", "tanh"],
                "regressor__alpha": [0.0001, 0.001],},),}


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
                f"{name} -> RMSE: {rmse:.4f}, R2: {r2:.4f}, Training Time: {elapsed_time:.2f} sec"
            )

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

