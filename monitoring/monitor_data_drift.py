import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
import os

# --- Configuration ---
# Define paths to your reference (training) and current (production) data
# Make sure these paths are correct for your project
REF_DATA_PATH = "data/company_sales_data.csv"
PROD_DATA_PATH = "data/cleaned/cleaned_data_latest.csv" # Update to your latest file
P_VALUE_THRESHOLD = 0.05 # Standard significance level

# --- Functions for Drift Detection ---
def detect_drift_in_numerical_feature(ref_col: pd.Series, prod_col: pd.Series, feature_name: str):
    """Performs a Kolmogorov-Smirnov test for numerical data drift."""
    # The KS test checks if two samples come from the same distribution
    statistic, p_value = ks_2samp(ref_col, prod_col)
    
    print(f"\n--- {feature_name} (Numerical) ---")
    print(f"KS Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < P_VALUE_THRESHOLD:
        print("🔴 Significant drift detected!")
    else:
        print("🟢 No significant drift detected.")


def detect_drift_in_categorical_feature(ref_col: pd.Series, prod_col: pd.Series, feature_name: str):
    """Performs a Chi-Squared test for categorical data drift."""
    # A Chi-Squared test compares observed and expected frequencies of categories
    ref_counts = ref_col.value_counts()
    prod_counts = prod_col.value_counts()

    # Ensure both series have the same categories to compare
    combined_index = ref_counts.index.union(prod_counts.index)
    ref_counts = ref_counts.reindex(combined_index, fill_value=0)
    prod_counts = prod_counts.reindex(combined_index, fill_value=0)

    chi2, p_value, _, _ = chi2_contingency([ref_counts, prod_counts])
    
    print(f"\n--- {feature_name} (Categorical) ---")
    print(f"Chi-Squared Statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < P_VALUE_THRESHOLD:
        print("🔴 Significant drift detected!")
    else:
        print("🟢 No significant drift detected.")


# Main Execution
if __name__ == "__main__":
    print("Starting manual data drift detection...")

    try:
        # Load the two datasets
        ref_data = pd.read_csv(REF_DATA_PATH)
        prod_data = pd.read_csv(PROD_DATA_PATH)
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure data files are in the correct location.")
        exit()

    # Define the features to monitor (numerical and categorical)

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



    # Check for drift in each numerical feature
    for feature in numerical_features:
        if feature in ref_data.columns and feature in prod_data.columns:
            detect_drift_in_numerical_feature(ref_data[feature], prod_data[feature], feature)
        else:
            print(f"Warning: Feature '{feature}' not found in one of the datasets.")

    # Check for drift in each categorical feature
    for feature in expanded_categorical_features:
        if feature in ref_data.columns and feature in prod_data.columns:
            detect_drift_in_categorical_feature(ref_data[feature], prod_data[feature], feature)
        else:
            print(f"Warning: Feature '{feature}' not found in one of the datasets.")
    
    print("\nData drift analysis complete.")