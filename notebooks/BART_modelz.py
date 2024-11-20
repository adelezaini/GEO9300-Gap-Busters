
import pandas as pd
import glob
import os
from scipy.stats import skew, normaltest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt


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

        # Get all hyperparameter results
    cv_results = grid_search.cv_results_

    return best_params, y_pred, cv_results



#--------------------------------------------------------------------------------

# BART:

# packages:
import sys
import os


# BART:

# packages:
import sys
import os

# OBS bartpy is a public GitHub repo so I cloned it locally to a folder on my computer and defined in below path where I cloned it to:
bartpy_path = '/home/mlahlbac/projects/BART/bartpy/'
sys.path.append(os.path.abspath(bartpy_path))
from bartpy.sklearnmodel import SklearnModel

# "true" y values for the test case:

y_true = syn_ds['LE'][LE_gaps['LE_gaps'].isnull()]


BART_model = SklearnModel(n_trees=50, n_burn=250, n_samples=1000)


model = SklearnModel(n_burn=50, n_chains=1, n_jobs=7, n_samples=50, n_trees=10)
model.fit(X_train, y_train)

# Evaluate the model on the gaps
y_pred = model.predict(X_test)


# model evaluation
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")



#--------------------------------------------------------------------------------
# Hyperparameter tuning:
param_grid_bart = {
    'n_trees': [50, 100],
    'n_burn': [200, 250],
    'n_samples': [50, 100]
}

try:
    BART_best_model, BART_y_pred = model_tuning_CV(BART_model, param_grid_bart, cv, scoring)
    BART_mse = mean_squared_error(y_true, BART_y_pred)
    BART_r2 = r2_score(y_true, BART_y_pred)
    print(f"BART - MSE: {BART_mse}, R2: {BART_r2}")
except Exception as e:
    print(f"An error occurred during BART tuning: {e}")