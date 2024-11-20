# Description: This script is used to create a workflow for the data analysis

import pandas as pd
import glob
import os
from scipy.stats import skew, normaltest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

# Script workflow:
# - load data
# - scaling
# - split into training and testing based on data gaps
# - machine learning models and applying hyperparameter search grids

################################################################################
### Workflow Parameters
################################################################################

# Define features to run as input variables for the models:
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

# Choose the gaps dataset - either structured or random gaps
gaps_data_file = 'structured_gaps_1' # 'random_gaps_1' -- values from 1 to 5 for diff versions

# Define the cross-validation strategy:
scoring = 'neg_mean_absolute_error'   # e.g. 'neg_mean_squared_error', 'r2', 'neg_root_mean_squared_error. More available methods for regression evaluation (scoring): https://scikit-learn.org/1.5/modules/model_evaluation.html#scoring-parameter)
cv = 3  # Number of cross-validation folds


################################################################################
### Data 
################################################################################

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
# Feautures and target variables:

syn_ds["time"] = pd.to_datetime(syn_ds["time"])
syn_ds["doy"] = syn_ds["time"].dt.dayofyear
syn_ds["hour"] = syn_ds["time"].dt.hour

y = syn_ds["LE"]
X = syn_ds[features]

#-------------------------------------------------------------------------------

# Apply scaling to variables:

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
        
    return scaled_df, scaling_info

#-------------------------------------------------------------------------------
# Split into training and testing datasets based on gaps in LE:

# Function to load data gaps dataset
def load_data_gaps(file_name):
    return pd.read_csv(f'../data/LE-gaps/{file_name}.csv', parse_dates=['time'], sep=',')

LE_gaps = load_data_gaps(gaps_data_file)

# select X and y where LE_gaps is not null
X_train = X[LE_gaps['LE_gaps'].notnull()]
y_train = y[LE_gaps['LE_gaps'].notnull()]

# The following test set is for where there is data gaps
X_test = X[LE_gaps['LE_gaps'].isnull()]
y_test = y[LE_gaps['LE_gaps'].isnull()]

#--------------------------------------------------------------------------------

# MACHINE LEARNING MODELS:

def model_tuning_CV(model, hyperparameters, cv, scoring):
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Parameters:
        model: The ML model to be tuned.
        hyperparameters: A grid of hyperparameters to be tuned.
        cv: Number of cross-validation folds.
        scoring: Scoring (evaluation) metric to be used.
    Returns:
        best_params: The best hyperparameters found by GridSearchCV.
        y_pred: Predictions made by the best model on the test set.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=hyperparameters, cv=cv, n_jobs=-1, scoring=scoring, verbose=2)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")
    print(f'Best Score: {grid_search.best_score_:.2f}')
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    return best_params, y_pred

#--------------------------------------------------------------------------------

# RANDOM FOREST REGRESSOR:

# packages:
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score  # Import regression metrics
from sklearn.model_selection import GridSearchCV # Import GridSearchCV for hyperparameter tuning

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 8]
}

RF_best_model, RF_y_pred = model_tuning_CV(RandomForestRegressor(), param_grid, cv, scoring)

#--------------------------------------------------------------------------------

# NEURAL NETWORK REGRESSOR:

# packages:
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from scikeras.wrappers import KerasRegressor


# Function to build the model -- the parameters are switched out to the ones in the grid search CV if specified.
def create_model(units=64, activation='relu'):
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(units, activation=activation, kernel_initializer='uniform'),
        Dropout(0.2),
        Dense(units, activation=activation),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

# Wrap the Keras model in a scikit-learn compatible model
keras_model = KerasRegressor(model=create_model, verbose=0, activation='relu', units=64)

# Create a matrix for hyperparameter tuning
param_grid = {
    'units': [32, 64, 128],
    'activation': ['relu', 'tanh', 'sigmoid'],   # we could also skip this param. and use the default value in the function?
    'batch_size': [10, 20, 30],
    'epochs': [50, 100, 200]
}

# Create the GridSearchCV object 
NN_best_model, NN_y_pred = model_tuning_CV(keras_model, param_grid, cv, scoring)

#--------------------------------------------------------------------------------
# XGBOOST REGRESSOR:

import xgboost as xgb

params = {
        "subsample" : [0.5, 0.75, 1],
        "n_estimators" : [50, 75, 100, 150],
        "max_depth" :  [3, 5, 10, 15],
        "min_child_weight" : [2, 5, 10],
        "colsample_bytree" : [0.4, 0.6, 0.8, 1],
    }

xgb_model = xgb.XGBRegressor()

XG_best_model, XG_y_pred = model_tuning_CV(xgb_model, param_grid, cv, scoring)

