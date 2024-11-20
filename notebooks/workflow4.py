# Description: This script is used to create a workflow for the data analysis

import pandas as pd
import glob
import os
import sys

# Get the parent directory and add it to sys.path
#parent_dir = os.path.abspath("/notebooks/workflow_Eivind")
#sys.path.insert(0, parent_dir)

# Now import the module
from workflow_Eivind.ML_lib_copy import * # NOTE: May need to change, depending on location.

# packages:
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import xgboost as xgb
from xgboost import XGBRegressor
#from bartpy.sklearnmodel import SklearnModel as BART

print("Imported libraries\n")

# Script workflow:
# - load data
# - scaling
# - split into training and testing based on data gaps
# - machine learning models and applying hyperparameter search grids

#### to change: import from external argument ###
algorithm = 'xgboost' #look below for the options
scoring_metrics = 'r2' #or 'mse'

print(f"Machine Learning method: {algorithm}")
print(f"Evaluation metrics: {scoring_metrics}")

# save hyperparameters for the run as csv file?
save_to_csv = True

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
CV_scoring = 'neg_mean_absolute_error'   # e.g. 'neg_mean_squared_error', 'r2', 'neg_root_mean_squared_error. More available methods for regression evaluation (scoring): https://scikit-learn.org/1.5/modules/model_evaluation.html#scoring-parameter)
cv = 3  # Number of cross-validation folds


################################################################################
### Data
################################################################################


##### Load the synthetic dataset:
# a. Load single CSV files in separate dfs
# b. Merge the dfs into one single "synthetic_dataset"

folder_path = '../data/synthetic_dataset' # NOTE: May need to adjust if the script is used from another folder

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

def split_train_test_dataset(original_X, original_y):
    # Function to load data gaps dataset
    def load_data_gaps(file_name):
        return pd.read_csv(f'../data/LE-gaps/{file_name}.csv', parse_dates=['time'], sep=',') # NOTE: May need to adjust if the script is used from another folder

    LE_gaps = load_data_gaps(gaps_data_file)

    # Select X and y where LE_gaps is not null
    X_train = original_X[LE_gaps['LE_gaps'].notnull()]
    y_train = original_y[LE_gaps['LE_gaps'].notnull()]

    # The following test set is for where there are data gaps
    X_test = original_X[LE_gaps['LE_gaps'].isnull()]

    # LE without gaps:
    LE = syn_ds['LE']
    # extract LE where LE_gaps is null:
    y_test = LE[LE_gaps['LE_gaps'].isnull()]

    print("Created training and testing datasets\n")
    return X_train, y_train, X_test, y_test

################################################################################
# MACHINE LEARNING MODELS
################################################################################

def machine_learning_method(algorithm):

    allowed_algorithms = {'linear_regression', 'random_forest', 'neural_network', 'lstm', 'xgboost', 'bart'}
    
    # Validate the algorithm
    assert algorithm in allowed_algorithms, (
        f"Invalid algorithm '{algorithm}'. "
        f"Allowed values are: {', '.join(allowed_algorithms)}."
    )
    
    if algorithm == 'linear_regression':

        # LINEAR REGRESSION (No hyperparameters to tune)
        
        # Apply scaling to input features
        X_scaled, _ = adaptive_scaling(X, cyclic_features = ['hour','doy'])
        
        # Split training and testing datasets
        X_train, y_train, X_test, y_test = split_train_test_dataset(X_scaled, y)
        
        # Apply machine learning method
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # Generate predictions
        y_pred = lr_model.predict(X_test)

        # Evaluate the ML method
        LR_metrics = evaluate_model(lr_model, X_test, y_test)

        print("\n=== LINEAR REGRESSION RESULTS ===")
        print(f"Test Metrics: {LR_metrics}")

        # Save predictions to CSV
        if save_to_csv:
            results_df = pd.DataFrame({
            'index': X_test.index,
            'Actual': y_test,
            'Predicted': y_pred
            })
            results_df.to_csv(f'../results/{algorithm}_{gaps_data_file}_predictions.csv', index=False)

        #-------------------------------------------------------------------------------
    elif algorithm == 'random_forest':

        # RANDOM FOREST REGRESSOR
        
        # Split training and testing datasets
        X_train, y_train, X_test, y_test = split_train_test_dataset(X, y)
        
        
        param_grid_rf = {
            'n_estimators': [100, 200, 300, 500],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8]
        }

        rf_model = RandomForestRegressor()
        RF_best_model, RF_best_params, cv_results = model_tuning_CV(X_train, y_train, rf_model, param_grid_rf, cv, CV_scoring)
        RF_metrics = evaluate_model(RF_best_model, X_test, y_test, scoring_metrics)

        # Generate predictions
        y_pred = RF_best_model.predict(X_test)

        print("\n=== RANDOM FOREST RESULTS ===")
        print(f"Best Parameters: {RF_best_params}")
        print(f"Test Metrics: {RF_metrics}")

        # save to csv
        if save_to_csv == True:
            cv_results_df = pd.DataFrame(cv_results)
            cv_results_df.to_csv(f'../results/{algorithm}_{gaps_data_file}_cv_results.csv', index=False)
            results_df = pd.DataFrame({
            'index': X_test.index,
            'Actual': y_test,
            'Predicted': y_pred
            })
            results_df.to_csv(f'../results/{algorithm}_{gaps_data_file}_predictions.csv', index=False)
        #-------------------------------------------------------------------------------
    elif algorithm == 'neural_network':
    
        # NEURAL NETWORK REGRESSOR
        
        # Apply scaling to input features
        X_scaled, _ = adaptive_scaling(X, cyclic_features = ['hour','doy'])
        
        # Split training and testing datasets
        X_train, y_train, X_test, y_test = split_train_test_dataset(X_scaled, y)
        
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

        NN_best_model, NN_best_params, cv_results = model_tuning_CV(X_train, y_train, keras_model, param_grid_nn, cv, CV_scoring)
        NN_metrics = evaluate_model(NN_best_model, X_test, y_test, scoring_metrics)

        # Generate predictions
        y_pred = NN_best_model.predict(X_test)

        print("\n=== NEURAL NETWORK RESULTS ===")
        print(f"Best Parameters: {NN_best_params}")
        print(f"Test Metrics: {NN_metrics}")
        # save to csv
        if save_to_csv == True:
            cv_results_df = pd.DataFrame(cv_results)
            cv_results_df.to_csv(f'../results/{algorithm}_{gaps_data_file}_cv_results.csv', index=False)
            results_df = pd.DataFrame({
            'index': X_test.index,
            'Actual': y_test,
            'Predicted': y_pred
            })
            results_df.to_csv(f'../results/{algorithm}_{gaps_data_file}_predictions.csv', index=False)
        #-------------------------------------------------------------------------------
    elif algorithm == 'lstm':
    
        # LSTM REGRESSOR
        
        # Apply scaling to input features
        X_scaled, _ = adaptive_scaling(X, cyclic_features = ['hour','doy'])
        
        # Split training and testing datasets
        X_train, y_train, X_test, y_test = split_train_test_dataset(X_scaled, y)
        
        
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

        LSTM_best_model, LSTM_best_params, cv_results = model_tuning_CV(X_train, y_train, lstm_model, param_grid_lstm, cv, CV_scoring)
        LSTM_metrics = evaluate_model(LSTM_best_model, X_test, y_test, scoring_metrics)

        # Generate predictions
        y_pred = LSTM_best_model.predict(X_test)

        print("\n=== LSTM RESULTS ===")
        print(f"Best Parameters: {LSTM_best_params}")
        print(f"Test Metrics: {LSTM_metrics}")
        # save to csv
        if save_to_csv == True:
            cv_results_df = pd.DataFrame(cv_results)
            cv_results_df.to_csv(f'../results/{algorithm}_{gaps_data_file}_cv_results.csv', index=False)
            results_df = pd.DataFrame({
            'index': X_test.index,
            'Actual': y_test,
            'Predicted': y_pred
            })
            results_df.to_csv(f'../results/{algorithm}_{gaps_data_file}_predictions.csv', index=False)

    #-------------------------------------------------------------------------------
    elif algorithm == 'xgboost':
    
        # XGBOOST REGRESSOR
        
        # Apply scaling to input features
        X_scaled, _ = adaptive_scaling(X, cyclic_features = ['hour','doy'])
        
        # Split training and testing datasets
        X_train, y_train, X_test, y_test = split_train_test_dataset(X_scaled, y)

        param_grid_xgb = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
        XGB_best_model, XGB_best_params, cv_results = model_tuning_CV(X_train, y_train, xgb_model, param_grid_xgb, cv, CV_scoring)
        XGB_metrics = evaluate_model(XGB_best_model, X_test, y_test, scoring_metrics)

        # Generate predictions
        y_pred = XGB_best_model.predict(X_test)

        print("\n=== XGBOOST RESULTS ===")
        print(f"Best Parameters: {XGB_best_params}")
        print(f"Test Metrics: {XGB_metrics}")
        # save to csv
        if save_to_csv == True:
            cv_results_df = pd.DataFrame(cv_results)
            cv_results_df.to_csv(f'../results/{algorithm}_{gaps_data_file}_cv_results.csv', index=False)
            results_df = pd.DataFrame({
            'index': X_test.index,
            'Actual': y_test,
            'Predicted': y_pred
            })
            results_df.to_csv(f'../results/{algorithm}_{gaps_data_file}_predictions.csv', index=False)
    #-------------------------------------------------------------------------------
    elif algorithm == 'bart':
    
        # BART REGRESSOR
        
        # Apply scaling to input features
        X_scaled, _ = adaptive_scaling(X, cyclic_features = ['hour','doy'])
        
        # Split training and testing datasets
        X_train, y_train, X_test, y_test = split_train_test_dataset(X_scaled, y)

        bart_model = BART(random_state=42)
        
        param_grid_bart = {
            'n_trees': [50, 100, 200],
            'alpha': [0.95, 0.99],
            'beta': [1.0, 2.0],
            'k': [2.0, 3.0]
        }

        BART_best_model, BART_best_params, cv_results = model_tuning_CV(X_train, y_train, bart_model, param_grid_bart, cv, CV_scoring)
        BART_metrics = evaluate_model(BART_best_model, X_test, y_test, scoring_metrics)

        # Generate predictions
        y_pred = bart_model.predict(X_test)

        print("\n=== BART RESULTS ===")
        print(f"Best Parameters: {BART_best_params}")
        print(f"Test Metrics: {BART_metrics}")
        # save to csv
        if save_to_csv == True:
            cv_results_df = pd.DataFrame(cv_results)
            cv_results_df.to_csv(f'../results/{algorithm}_{gaps_data_file}_cv_results.csv', index=False)
            results_df = pd.DataFrame({
            'index': X_test.index,
            'Actual': y_test,
            'Predicted': y_pred
            })
            results_df.to_csv(f'../results/{algorithm}_{gaps_data_file}_predictions.csv', index=False)

# Perform the function
machine_learning_method(algorithm)