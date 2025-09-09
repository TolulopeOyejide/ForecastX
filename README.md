# ForecastX – Company Sales Prediction App
ForecastX is a complete MLOps pipeline designed to generate accurate sales forecasts. It ingests historical products(CSV/Excel) and delivers actionable predictions of future sales for individual customers. This solution enables businesses to proactively manage inventory, optimize marketing strategies, and improve overall decision-making.


## The Demo Video

Here's a quick demo video that shows Forecast - Company Sales Prediction App in action.

[**Watch the Demo Video**](https://www.youtube.com/watch?v=3u1AGHt1flA)



## Features

-   **Data Management**:
    -   Handles data cleaning and preprocessing.
    -   Manages data and pipeline versions using DVC.

-   **Automated Monitoring & Tracking**:
    -   Includes automated data drift monitoring using statistical methods.
    -   Tracks experiments and manages model lineage with MLflow.

-   **Advanced Model Development**:
    -   Features an automated Grid Search CV pipeline for robust model training.
    -   Explores a comprehensive suite of regression algorithms to select the best forming:
        -   _Linear Regression_: A simple, interpretable model that finds the best-fitting straight line to predict a continuous outcome.
        -   _Polynomial Regression_: An extension of linear regression that fits a curved line to capture non-linear relationships in the data.
        -   _Random Forest Regressor_: An ensemble model that uses multiple decision trees to make robust predictions, reducing the risk of overfitting.
        -   _XGBoost Regressor_: A powerful, high-performance gradient boosting model known for its accuracy and efficiency.
        -   _K-Nearest Neighbors (KNN) Regressor_: A simple, instance-based model that predicts a value by averaging the values of the "k" most similar data points.
        -   _Support Vector Regressor (SVR)_: An effective model for high-dimensional data that finds a function to fit the data within a specified margin of tolerance.
        -   _Neural Network Regressor_: A complex, brain-inspired model capable of learning highly non-linear and intricate patterns for precise predictions.

-   **Production & MLOps**:
    -   Serves predictions through a scalable API built with FastAPI.
    -   Uses Docker for containerized deployment, ensuring portability.
    -   Automates the entire workflow with a Continuous Integration/Continuous Deployment (CI/CD) pipeline using GitHub Actions.



## Project Structure
```
ForecastX/
│
├── data/                     # Raw and processed datasets
├── src/                      # Source code for the pipeline
│   ├── process_data.py       # Cleans and preprocesses data
│   ├── train_log_models.py   # Train and log model
│   └── load_model.py         # Supply the best trained and latest model to API
├── models/                   # Saved models
├── app/                      # Source code for the application
│   ├── main.py               # Serving the model with Fast API
│   └── app.py                # Streamlit UI of the ForecastX App
├── monitoring                # Data Drift Monitoring
│   ├── monitor_data_drift.py # Track Data Drift 
├── .github/workflows         # CI/CD 
│   ├── jobs.yml              # Automating the MLOPs pipeline run
├── Dockerfile                # Docker setup
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```


## Setup Instructions
1. Clone the repository:  <br>
   `https://github.com/TolulopeOyejide/ForecastX.git`



2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


3. Test the model API:  <br>
  `uvicorn app.main:app --host 0.0.0.0 --port 8005` <br>


4. Run the Docker <br>

   You can use the pre-built Docker image to get the API running instantly without building it yourself.  <br>

   The Docker image for this project is available on [Docker Hub](https://hub.docker.com/repository/docker/tolulopeoyejide/sales-prediction-api). <br>

   To pull the image and run the container: <br>
   `docker pull tolulopeoyejide/sales-prediction-api` <br>
   `docker run -d -p 8002:8002 --name sales-api-container tolulopeoyejide/sales-prediction-api`

