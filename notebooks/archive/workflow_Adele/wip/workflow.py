# Description: This script is used to create a workflow for the data analysis

import pandas as pd
import glob
import os
from ML_lib import *


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


#--------------------------------------------------------------------------------

# RANDOM FOREST REGRESSOR:

# packages:
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_squared_error, r2_score  # Import regression metrics

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

