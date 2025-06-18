import pandas as pd
import numpy as np
from scipy.stats import skew, normaltest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

def adaptive_scaling(dataframe, target_column=None, scaling_method="individual", verbose = False, plot = False):
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
        
        if verbose: print(f"Processing column: {col}")
        
        # Check for missing values
        if scaled_df[col].isnull().any():
            if verbose: print(f"Warning: Column '{col}' contains missing values. Imputing with median.")
            scaled_df[col].fillna(scaled_df[col].median(), inplace=True)

        # Decide scaling method
        if scaling_method == "individual":
            skewness = skew(scaled_df[col])
            _, p_value = normaltest(scaled_df[col])
            if verbose: print(f"  Skewness: {skewness:.2f}, Normality test p-value: {p_value:.4f}")
            
            if p_value > 0.05:  # Normally distributed
                message = f"  Applying StandardScaler (data is approximately normal)."
                scaler = StandardScaler()
            elif skewness > 1 or skewness < -1:  # Highly skewed
                message = f"  Applying log transformation followed by MinMaxScaler (data is skewed)."
                scaled_df[col] = np.log1p(scaled_df[col] - scaled_df[col].min() + 1)
                scaler = MinMaxScaler()
            else:  # Mildly skewed or uniform
                message = f"  Applying MinMaxScaler (data is mildly skewed or uniform)."
                scaler = MinMaxScaler()
        elif scaling_method == "minmax":
            message = f"  Applying MinMaxScaler (user-specified method)."
            scaler = MinMaxScaler()
        elif scaling_method == "standard":
            message = f"  Applying StandardScaler (user-specified method)."
            scaler = StandardScaler()
        elif scaling_method == "logminmax":
            message = f"  Applying log transformation followed by MinMaxScaler (user-specified method)."
            scaled_df[col] = np.log1p(scaled_df[col] - scaled_df[col].min() + 1)
            scaler = MinMaxScaler()
            
        if verbose: print(message)
        
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
        
        if plot:
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
  
