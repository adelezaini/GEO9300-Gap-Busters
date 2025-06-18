# Description: This script is used to create a workflow for the data analysis

import pandas as pd
import glob
import os
from ML_lib import *  # Assuming your tuning and evaluation functions are here

# packages:
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import xgboost as xgb
#from bartpy.sklearnmodel import SklearnModel as BART

print("Imported libraries\n")

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
    "SWin",
    "LWin",
    "Tair",
    "RH_air",
    "prec",
    "u",
    "snow_cover",
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


##### Load the synthetic dataset:
# a. Load single CSV files in separate dfs
# b. Merge the dfs into one single "synthetic_dataset"

folder_path = '../../data/synthetic_dataset'

csv_files = glob.glob(os.path.join(folder_path, '*.csv')) # use glob library to find all CSV files

dfs = [] #to store individual DataFrames.

for file in csv_files:
    data = pd.read_csv(file, parse_dates=['time'], sep=',')
    # 'parse_dates' argument ensures the 'time' column is interpreted as datetime objects.
    
    dfs.append(data)

syn_ds = dfs[0] # Start with the first DataFrame as the base for merging.

for data in dfs[1:]:
    # Merge each subsequent DataFrame with the base DataFrame (`syn_ds`).
    # The merge is done using an ordered merge on the 'time' column.
    # This ensures that the merged dataset remains sorted by 'time'.
    syn_ds = pd.merge_ordered(syn_ds, data, on='time')

#-------------------------------------------------------------------------------
# Features and target variables:

syn_ds["time"] = pd.to_datetime(syn_ds["time"])
syn_ds["doy"] = syn_ds["time"].dt.dayofyear
syn_ds["hour"] = syn_ds["time"].dt.hour

y = syn_ds["LE"]
X = syn_ds[features]

#-------------------------------------------------------------------------------
# Split into training and testing datasets based on gaps in LE:

# Function to load data gaps dataset
def load_data_gaps(file_name):
    return pd.read_csv(f'../data/LE-gaps/{file_name}.csv', parse_dates=['time'], sep=',')

LE_gaps = load_data_gaps(gaps_data_file)

# Select X and y where LE_gaps is not null
X_train = X[LE_gaps['LE_gaps'].notnull()]
y_train = y[LE_gaps['LE_gaps'].notnull()]

# The following test set is for where there are data gaps
X_test = X[LE_gaps['LE_gaps'].isnull()]
y_test = y[LE_gaps['LE_gaps'].isnull()]

print("Created training and testing datasets")

################################################################################
# MACHINE LEARNING MODELS
################################################################################

#-------------------------------------------------------------------------------

# LINEAR REGRESSION (No hyperparameters to tune)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
LR_metrics = evaluate_model(lr_model, X_test, y_test, scoring)

print("\n=== LINEAR REGRESSION RESULTS ===")
print(f"Test Metrics: {LR_metrics}")

#-------------------------------------------------------------------------------

# BART REGRESSOR
bart_model = BART()
BART_best_model, BART_best_params = model_tuning_CV(X_train, y_train, bart_model, {}, cv, scoring)
BART_metrics = evaluate_model(BART_best_model, X_test, y_test, scoring)

print("\n=== BART RESULTS ===")
print(f"Best Parameters: {BART_best_params}")
print(f"Test Metrics: {BART_metrics}")

#-------------------------------------------------------------------------------

# RANDOM FOREST REGRESSOR
param_grid_rf = {
    'n_estimators': [100, 200, 300, 500],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 8]
}

rf_model = RandomForestRegressor()
RF_best_model, RF_best_params = model_tuning_CV(X_train, y_train, rf_model, param_grid_rf, cv, scoring)
RF_metrics = evaluate_model(RF_best_model, X_test, y_test, scoring)

print("\n=== RANDOM FOREST RESULTS ===")
print(f"Best Parameters: {RF_best_params}")
print(f"Test Metrics: {RF_metrics}")

#-------------------------------------------------------------------------------

# NEURAL NETWORK REGRESSOR
def create_NN_model(units=64, activation='relu'):
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

keras_model = KerasRegressor(model=create_NN_model, verbose=0)

param_grid_nn = {
    'model__units': [32, 64, 128],
    'model__activation': ['relu', 'tanh', 'sigmoid'],
    'batch_size': [10, 20, 30],
    'epochs': [50, 100, 200]
}

NN_best_model, NN_best_params = model_tuning_CV(X_train, y_train, keras_model, param_grid_nn, cv, scoring)
NN_metrics = evaluate_model(NN_best_model, X_test, y_test, scoring)

print("\n=== NEURAL NETWORK RESULTS ===")
print(f"Best Parameters: {NN_best_params}")
print(f"Test Metrics: {NN_metrics}")

#-------------------------------------------------------------------------------

# LSTM REGRESSOR
def create_lstm_model(units=64, activation='relu'):
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(units, activation=activation, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

lstm_model = KerasRegressor(model=create_lstm_model, verbose=0)

param_grid_lstm = {
    'model__units': [32, 64, 128],
    'model__activation': ['relu', 'tanh'],
    'batch_size': [10, 20],
    'epochs': [50, 100]
}

LSTM_best_model, LSTM_best_params = model_tuning_CV(X_train, y_train, lstm_model, param_grid_lstm, cv, scoring)
LSTM_metrics = evaluate_model(LSTM_best_model, X_test, y_test, scoring)

print("\n=== LSTM RESULTS ===")
print(f"Best Parameters: {LSTM_best_params}")
print(f"Test Metrics: {LSTM_metrics}")

#-------------------------------------------------------------------------------

# XGBOOST REGRESSOR
param_grid_xgb = {
    "subsample": [0.5, 0.75, 1],
    "n_estimators": [50, 75, 100, 150],
    "max_depth": [3, 5, 10, 15],
    "min_child_weight": [2, 5, 10],
    "colsample_bytree": [0.4, 0.6, 0.8, 1]
}

xgb_model = xgb.XGBRegressor()
XG_best_model, XG_best_params = model_tuning_CV(X_train, y_train, xgb_model, param_grid_xgb, cv, scoring)
XG_metrics = evaluate_model(XG_best_model, X_test, y_test, scoring)

print("\n=== XGBOOST RESULTS ===")
print(f"Best Parameters: {XG_best_params}")
print(f"Test Metrics: {XG_metrics}")
