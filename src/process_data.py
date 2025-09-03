import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
import subprocess
import os
from datetime import datetime


RAW_DATA_PATH = "data/company_sales_data.csv" #adjustable file name
CLEAN_DATA_DIR = "data/cleaned/"

if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError("data not found. Please upload your raw dataset first.")

# Create cleaned folder if it doesn't exist
os.makedirs(CLEAN_DATA_DIR, exist_ok=True)

# Read raw data
df = pd.read_csv(RAW_DATA_PATH, encoding="ISO-8859-1")
print(f"Raw Data Shape: {df.shape}")



#----Statutory Data Quality Checks & Fixes

# 1. Standardize the column headers
def clean_column_names(df):
    """
    Make column headers lowercase and remove whitespace for consistency.
    """
    df.columns = (
        df.columns.str.strip()      # remove leading/trailing spaces
                  .str.lower()      # make lowercase
                  .str.replace(" ", "_")  # replace spaces with underscore
    )
    return df





# 2.Check data types of columns.
def check_dtypes(df):
    """
    Returns the data types of all columns.
    """
    return df.dtypes




# 3. Check & Handle Missing Values.
def check_missing_values(df):
    """
    Returns a DataFrame with counts and percentages of missing values per column.
    """
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    return pd.DataFrame({'missing_count': missing, 'missing_percent': percent}).sort_values(by='missing_count', ascending=False)




def handle_missing_values(df, strategy="mean", fill_value=None):
    """
    Handle missing values based on chosen strategy.
    strategy options: 'mean', 'median', 'mode', 'drop', 'constant'
    """
    if strategy == "mean":
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == "median":
        return df.fillna(df.median(numeric_only=True))
    elif strategy == "mode":
        return df.fillna(df.mode().iloc[0])
    elif strategy == "drop":
        return df.dropna()
    elif strategy == "constant":
        return df.fillna(fill_value)
    else:
        raise ValueError("Invalid strategy. Choose from: mean, median, mode, drop, constant.")   




# 4. Check and Handle Duplicates.
def check_duplicates(df):
    """
    Returns the number of duplicate rows in the DataFrame.
    """
    return df.duplicated().sum()


def remove_duplicates(df):
    """Remove duplicate rows."""
    return df.drop_duplicates()





# 5. Check and Handle Outliers.
def check_outliers(df, cols=None):
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns
    
    outlier_info = []
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_info.append({"column": col, "outlier_count": outliers})
    
    return pd.DataFrame(outlier_info)





def handle_outliers(df, column = None):
    """Clip outliers in a column using IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df








# 6. Check feature_target correlations (Use in Training and Logging Python File)
def check_feature_target_correlation(df, target, threshold=0.4):
    """
    Returns features that have correlation with the target above the threshold.
    Also stores the selected features in a list.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        target (str): Target column name
        threshold (float): Minimum absolute correlation with target to consider
    
    Returns:
        tuple:
            pd.DataFrame: Sorted features and their correlation with target
            list: List of selected features
    """
    # Ensure target is in dataframe
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    # Compute correlations
    corr_matrix = df.corr(numeric_only=True)
    
    if target not in corr_matrix:
        raise ValueError(f"Target '{target}' is not numeric. Try encoding it first.")
    
    target_corr = corr_matrix[target].drop(target)  # drop self-correlation
    
    # Filter based on threshold
    selected = target_corr[abs(target_corr) >= threshold]
    
    # Sort by absolute correlation
    selected_sorted = selected.sort_values(key=abs, ascending=False).to_frame("correlation")
    
    # Store correlated feature names in a list
    features = selected_sorted.index.tolist()
    
    return selected_sorted, features







#7. Check Variable Relationships (Use in Training and Logging Python File)
def check_variable_relationships(df, target, top_n=5, plot=True):
    """
    Check linear or non-linear relationships between features and target.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing features and target
    target : str
        Target column name
    top_n : int, default=5
        Number of top features to plot based on correlation strength
    plot : bool, default=True
        Whether to plot scatterplots for visualization

    Returns
    -------
    results : pd.DataFrame
        DataFrame showing Pearson and Spearman correlations
    """

    results = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target:
            continue

        # Pearson for linearity
        pearson_corr = df[col].corr(df[target], method="pearson")

        # Spearman for monotonic non-linear
        spearman_corr, _ = spearmanr(df[col], df[target])

        results.append({
            "feature": col,
            "pearson_corr": pearson_corr,
            "spearman_corr": spearman_corr,
            "likely_relation": (
                "linear" if abs(pearson_corr) > abs(spearman_corr)
                else "non-linear/monotonic"
            )
        })

    results_df = pd.DataFrame(results).sort_values(
        by=["spearman_corr"], key=abs, ascending=False
    )

    if plot:
        top_features = results_df.head(top_n)["feature"]
        for col in top_features:
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=df[col], y=df[target], alpha=0.6)
            plt.title(f"{col} vs {target}")
            plt.show()

    return results_df





#8. Check if the feature needs scaling

def check_feature_scaling(df):
    """
    Check which features may need scaling.
    Returns:
        dict with keys:
          - 'stats': summary statistics (min, max, mean, std)
          - 'needs_scaling': list of columns needing scaling
    """
    numeric_df = df.select_dtypes(include=["number"])  # only numeric features
    desc = numeric_df.describe().T  # get summary stats
    needs_scaling = []

    for col in numeric_df.columns:
        col_range = desc.loc[col, "max"] - desc.loc[col, "min"]
        if col_range > 10:  # heuristic: large range means scaling is useful
            needs_scaling.append(col)

    return {
        "stats": desc,
        "needs_scaling": needs_scaling
    }



def scale_features(df, method="standard"):
    """
    Scale numeric features in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        method (str): 'standard' for StandardScaler, 'minmax' for MinMaxScaler.
    
    Returns:
        pd.DataFrame: DataFrame with scaled numeric features.
    """
    # Copy to avoid changing original
    df_scaled = df.copy()
    
    # Select numeric columns
    numeric_cols = df_scaled.select_dtypes(include=['number']).columns
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")
    
    # Fit & transform
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    df = df_scaled
    
    return df




#9. Encode Categorical Columns.


def encode_categorical(df, onehot_cols=None, label_cols=None):
    """
    Encode categorical columns in a DataFrame with flexibility.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        onehot_cols (list): Columns to one-hot encode
        label_cols (list): Columns to label encode
    
    Returns:
        pd.DataFrame: DataFrame with encoded categorical columns
    """
    df_encoded = df.copy()
    
    # One-hot encoding (order does not matter, dummy variables)
    if onehot_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=onehot_cols, drop_first=True)
    
    # Label encoding (order matters)
    if label_cols:
        le = LabelEncoder()
        for col in label_cols:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    df = df_encoded
    
    return df



# 10.Check class imbalance (for classification problems) 
def check_class_balance(df, target):  #Use in Training and Logging Python File
    """
    Returns the distribution of classes in the target column.
    """
    return df[target].value_counts(normalize=True) * 100



def handle_class_imbalance(X, y, method="smote"):
    """
    Handle class imbalance using different resampling techniques.

    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target vector
    method : str, default="smote"
        Options:
        - "smote" : SMOTE (Synthetic Minority Oversampling Technique) → generate synthetic samples of minority class.
        - "oversample" :Random Oversampling → duplicate minority class samples.
        - "undersample" : Random Undersampling → remove samples from majority class.

    Returns:
    --------
    X_res, y_res : Resampled feature matrix and target vector
    """
    if method == "smote":
        sampler = SMOTE(random_state=42)
    elif method == "oversample":
        sampler = RandomOverSampler(random_state=42)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError("method must be one of ['smote', 'oversample', 'undersample']")

    X_res, y_res = sampler.fit_resample(X, y)

    X_res, y_res  = X, y

    return X, y



#BEGIN DATA PREPROCESSING AND ANALYSIS

def preprocess_data_linear_reg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a full data preprocessing for linear regression models,
    including data cleaning, outlier handling, and scaling.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The processed DataFrame ready for model training.
    """
    print("\n--- Starting The Preprocessing Pipeline ---")

    # Step 1: Standardize the column headers
    df = clean_column_names(df)

    # Step 2: Check and handle missing values
    missing = check_missing_values(df)
    if missing is not None and not missing.empty:
        print("\nHandling missing values...")
        df = handle_missing_values(df, strategy="mean") #adjustable arguement
    else:
        print("\nNo missing values found.")

    # Step 3: Check and handle duplicates
    duplicates = check_duplicates(df)
    if duplicates is not None and duplicates > 0:
        print(f"Found {duplicates} duplicate rows. Removing...")
        df = remove_duplicates(df)
    else:
        print("\nNo duplicates found.")

    # Step 4: Check and handle outliers for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    for col in numeric_cols:
        outliers = check_outliers(df, cols=[col])
        if outliers is not None and not outliers.empty:
            print(f"\nHandling outliers in column: {col}")
            df = handle_outliers(df, column=col)
        else:
            print(f"No outliers found in column: {col}")

    # Step 5: Encode Categorical Columns
    print("\n--- Encoding Categorical Columns ---")
    df = encode_categorical(df, onehot_cols=['productline', 'productcode', 'customername', 'country']) #adjustable argument

    # Step 6: Scale numerical features
    # Note: This is an important step for linear regression.
    # We check for scaling needs and apply it only if necessary.
    scaling_info = check_feature_scaling(df)
    if scaling_info["needs_scaling"]:
        print("\nScaling numeric features...")
        df = scale_features(df, method="standard")
    else:
        print("\nNo scaling needed.")

    print("\n--- Data Preprocessing Complete ---")
    print(f"Final DataFrame shape: {df.shape}")
    print("Final Columns:", df.columns.tolist())
    return df


if __name__ == "__main__":
    preprocess_data_linear_reg(df)




#Save Processed Data
def save_processed_data(df: pd.DataFrame, clean_data_dir: str = "data/cleaned"):
    """
    Saves a processed DataFrame to a specified directory with a timestamp
    and also saves a latest copy to a different location.

    Args:
        df (pd.DataFrame): The processed DataFrame to be saved.
        clean_data_dir (str): The directory to save the timestamped file.
    """
    # Create the directory if it doesn't exist
    os.makedirs(clean_data_dir, exist_ok=True)
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(clean_data_dir, f"ref_data_{timestamp}.csv")
    df.to_csv(save_path, index=False)
    print(f"Cleaned data saved to {save_path}")

    # Save a latest copy for MLflow/training
    latest_path = os.path.join("data", "ref_data.csv")
    df.to_csv(latest_path, index=False)
    print(f"Latest clean data also saved to {latest_path}")



if __name__ == "__main__":
    save_processed_data(df)



#Data Versioning
def integrate_dvc():
    """Run DVC integration commands for tracking training data (idempotent)."""
    commands = []

    # Only initialize git if not already a repo
    if not os.path.exists(".git"):
        commands.append("git init")

    # Use -f to avoid failure if DVC is already initialized
    commands.append("dvc init -f")

    # Commit initialization (ignore failure if already done)
    commands.append('git commit -m "Initialize DVC" || echo "DVC already initialized"')

    # Track dataset
    commands.append("dvc add data/ref_data.csv")

    # Add .gitignore only if it exists
    if os.path.exists(".gitignore"):
        commands.append("git add data/ref_data.csv.dvc .gitignore")
    else:
        commands.append("git add data/ref_data.csv.dvc")

    commands.append('git commit -m "Track cleaned training data with DVC" || echo "Already tracked"')

    # Execute each command once
    for cmd in commands:
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=False)


if __name__ == "__main__":
    integrate_dvc()
