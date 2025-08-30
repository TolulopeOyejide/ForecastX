import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import subprocess
import os
from datetime import datetime


RAW_DATA_PATH = "data/sample_data.csv"
CLEAN_DATA_DIR = "data/cleaned/"

if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError("data not found. Please upload your raw dataset first.")

# Create cleaned folder if it doesn't exist
os.makedirs(CLEAN_DATA_DIR, exist_ok=True)

# Read raw data
df = pd.read_csv(RAW_DATA_PATH)
print(f"Raw Data Shape: {df.shape}")


def create_time_features(df, date_col='date'):
    """
    Create time-based features from a given date column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing at least a date column.
    date_col : str, optional (default='date')
        Column name to parse as datetime.

    Returns
    -------
    pandas.DataFrame
        Dataframe with new time features: day, weekday, month.
    """
    # Ensure the date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract day of the month
    df['day'] = df[date_col].dt.day
    
    # Extract day of the week (0 = Monday, 6 = Sunday)
    df['weekday'] = df[date_col].dt.weekday
    
    # Extract month number (1-12)
    df['month'] = df[date_col].dt.month
    
    return df


def aggregate_daily(df):
    """
    Aggregate sales and related features at the daily-product level.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing 'date', 'product_id', 'sales', 
        'price', 'promo', and 'marketing_spend' columns.

    Returns
    -------
    pandas.DataFrame
        Aggregated dataframe with sum/mean/max operations.
    """
    # Group by date and product, applying aggregation functions
    agg = df.groupby(['date', 'product_id']).agg({
        'sales': 'sum',             # Total daily sales per product
        'price': 'mean',            # Average daily price per product
        'promo': 'max',             # Whether promo was active that day
        'marketing_spend': 'sum'    # Total marketing spend
    }).reset_index()
    
    return agg


def add_rolling_features(df, window=7):
    """
    Add rolling statistical features to capture sales trends.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing 'date', 'product_id', and 'sales'.
    window : int, optional (default=7)
        Rolling window size (number of days).

    Returns
    -------
    pandas.DataFrame
        Dataframe with new rolling mean and standard deviation features.
    """
    # Ensure chronological order within each product
    df = df.sort_values(['product_id', 'date'])
    
    # Rolling mean of sales over the specified window
    df['sales_roll_mean_7'] = (
        df.groupby('product_id')['sales']
          .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    
    # Rolling standard deviation of sales (fill NaN with 0 for stability)
    df['sales_roll_std_7'] = (
        df.groupby('product_id')['sales']
          .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
    )
    
    return df



#----Statutory Data Quality Checks & Fixes

# 1.Check data types of columns.
def check_dtypes(df):
    """
    Returns the data types of all columns.
    """
    return df.dtypes





# 2. Check & Handle Missing Values.
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




# 3. Check and Handle Duplicates.
def check_duplicates(df):
    """
    Returns the number of duplicate rows in the DataFrame.
    """
    return df.duplicated().sum()


def remove_duplicates(df):
    """Remove duplicate rows."""
    return df.drop_duplicates()





# 4. Check and Handle Outliers.
def check_outliers(df, cols=None):
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns
    outlier_info = {}
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_info[col] = outliers
    return outlier_info




def handle_outliers(df, column = None):
    """Clip outliers in a column using IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df








# 5. Check feature_target correlations (Use in Training and Logging Python File)
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







#6. Check Variable Relationships (Use in Training and Logging Python File)
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





#7. Check if the feature needs scaling

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





# 8.Check class imbalance (for classification problems) 
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
    return X_res, y_res





#BEGIN DATA PREPROCESSING AND ANALYSIS
# Step 1: create_time_features
df = create_time_features(df)


# Step 2: aggregate_daily
df = aggregate_daily(df)


#Step 3: add rolling features
df = add_rolling_features(df)


#Step 4: check data types
print("\n--- Data Types ---")
print(check_dtypes(df))


#Step 5: Check and Fix Mising Values
print("\n--- Missing Values ---")
print(check_missing_values(df))
# Optionally handle missing values
df = handle_missing_values(df, strategy="mean")


#Step 6:print("\n--- Duplicate Count ---")
print(check_duplicates(df))
df = remove_duplicates(df)



#Step 7: Check and fix outliers
print("\n--- Outliers ---")
print(check_outliers(df, cols=["sales", "price"]))   #adjustable columns
# Example: handle outliers in one column
df = handle_outliers(df, column="sales")


# Step 8:Feature-Target Correlations
print("\n--- Feature Correlations with Target (sales) ---")
correlations, features = check_feature_target_correlation(df, target="sales") #adjustable columns
print(correlations)
print("Selected Features:", features)



# Step 9: Check variable relationships
print("\n--- Variable Relationships ---")
relationships = check_variable_relationships(df, target="sales", plot=False) #adjustable columns
print(relationships.head())


#Step 10: Scaling Check
print("\n--- Scaling Check ---")
scaling_info = check_feature_scaling(df)
print(scaling_info["stats"])
print("Needs scaling:", scaling_info["needs_scaling"])









 # Save with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_path = os.path.join(CLEAN_DATA_DIR, f"ref_data_{timestamp}.csv")
df.to_csv(save_path, index=False)
print(f"Cleaned data saved to {save_path}")


#Save a latest copy for MLflow/training
latest_path = os.path.join("data", "ref_data.csv")
df.to_csv(latest_path, index=False)
print(f"Latest clean data also saved to {latest_path}")


#Data Versioning
def integrate_dvc():
    """Run DVC integration commands for tracking training data."""
    commands = [
        "git init",
        "dvc init",
        'git commit -m "Initialize DVC"',
        "dvc add data/ref_data.csv",
        "git add data/ref_data.csv.dvc .gitignore",
        'git commit -m "Track cleaned training data with DVC"'
    ]

    for cmd in commands:
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    integrate_dvc()
