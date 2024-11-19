# Description: This script is used to create a workflow for the data analysis

import pandas as pd
import glob
import os

#-------------------------------------------------------------------------------
# Load the synthetic dataset:

folder_path = '../data/synthetic_dataset'
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

dfs = []
for file in csv_files:
    data = pd.read_csv(file, parse_dates=['time'], sep=',')
    dfs.append(data)

syn_ds = dfs[0]
for data in dfs[1:]:
    syn_ds = pd.merge_ordered(syn_ds, data, on='time')

print(syn_ds.head())

#-------------------------------------------------------------------------------
# Feauture selection:

# Get the DOY from the timestamp:
syn_ds["time"] = pd.to_datetime(syn_ds["time"])
syn_ds["doy"] = syn_ds["time"].dt.dayofyear

# Get the hour of the day from the timestamp:
syn_ds["hour"] = syn_ds["time"].dt.hour

# Define features and target variable:
features = [
    "u", 
    "LWin", 
    "SWin",
    "Tair", 
    "prec", 
    "snow_cover",
    "RH_air",
    "hour",
    "doy"
    ]

y = syn_ds["LE"]
X = syn_ds[features]

#-------------------------------------------------------------------------------

# Apply scaling here:
import pandas as pd
import numpy as np
from scipy.stats import skew, normaltest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

# Define the function
def adaptive_scaling(dataframe, target_column=None, scaling_method="minmax"):
    """
    Scales the features in the dataframe based on the specified or adaptive scaling method.
    
    Parameters:
        dataframe (pd.DataFrame): Input dataset.
        target_column (str): Name of the target column to exclude from scaling.
        scaling_method (str): Scaling method to use. Options are:
            - "individual": Decide scaling per column based on distribution.
            - "minmax": Use MinMaxScaler for all features.
            - "standard": Use StandardScaler for all features.
            - "logminmax": Apply log transformation followed by MinMaxScaler for all features.
    Returns:
        scaled_df (pd.DataFrame): Scaled dataset with the same columns as the input.
        scaling_info (dict): Information about the scaling method used for each column.
    """
    assert scaling_method in ["individual", "minmax", "standard", "logminmax"], \
        "Invalid scaling_method. Choose from 'individual', 'minmax', 'standard', 'logminmax'."
    
    scaled_df = dataframe.copy()
    scaling_info = {}
    columns = [col for col in dataframe.columns if col != target_column]

    for col in columns:
        
        print(f"Processing column: {col}")
        
        # Check for missing values
        if scaled_df[col].isnull().any():
            print(f"Warning: Column '{col}' contains missing values. Imputing with median.")
            scaled_df[col].fillna(scaled_df[col].median(), inplace=True)

        # Decide scaling method
        if scaling_method == "individual":
            skewness = skew(scaled_df[col])
            _, p_value = normaltest(scaled_df[col])
            print(f"  Skewness: {skewness:.2f}, Normality test p-value: {p_value:.4f}")
            
            if p_value > 0.05:  # Normally distributed
                print(f"  Applying StandardScaler (data is approximately normal).")
                scaler = StandardScaler()
            elif skewness > 1 or skewness < -1:  # Highly skewed
                print(f"  Applying log transformation followed by MinMaxScaler (data is skewed).")
                scaled_df[col] = np.log1p(scaled_df[col] - scaled_df[col].min() + 1)
                scaler = MinMaxScaler()
            else:  # Mildly skewed or uniform
                print(f"  Applying MinMaxScaler (data is mildly skewed or uniform).")
                scaler = MinMaxScaler()
        elif scaling_method == "minmax":
            print(f"  Applying MinMaxScaler (user-specified method).")
            scaler = MinMaxScaler()
        elif scaling_method == "standard":
            print(f"  Applying StandardScaler (user-specified method).")
            scaler = StandardScaler()
        elif scaling_method == "logminmax":
            print(f"  Applying log transformation followed by MinMaxScaler (user-specified method).")
            scaled_df[col] = np.log1p(scaled_df[col] - scaled_df[col].min() + 1)
            scaler = MinMaxScaler()
        
        # Apply scaling
        scaled_values = scaler.fit_transform(scaled_df[col].values.reshape(-1, 1))
        scaled_df[col] = scaled_values.flatten()
        scaling_info[col] = {
            "method": scaling_method if scaling_method != "individual" else type(scaler).__name__,
        }
        
        if scaling_method == "individual":
            scaling_info[col].update({
                "skewness": skewness,
                "normality_p_value": p_value,
            })
        
        plt.figure(figsize=(12, 6))
        
        # Plot non-scaled distribution
        plt.subplot(1, 2, 1)
        plt.hist(dataframe[col], bins=30, alpha=0.7, label='Non-Scaled')
        plt.title(f'Non-Scaled Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.legend()
        
        # Plot scaled distribution
        plt.subplot(1, 2, 2)
        plt.hist(scaled_df[col], bins=30, alpha=0.7, label='Scaled')
        plt.title(f'Scaled Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    return scaled_df, scaling_info


#-------------------------------------------------------------------------------
# Train-test split -- first import the data gaps ds:

# LE_gaps_ds = pd.read_csv('../data/synthetic_dataset_with_gaps.csv', parse_dates=['time'], sep=',')

# X_train
# Y_train

# The following test set is for where there is data gaps:
# X_test
# Y_test

#--------------------------------------------------------------------------------

# Apply machine learning model here:


