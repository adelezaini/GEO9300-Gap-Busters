import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, normaltest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


#------------------------- Scaling: ------------------------------#
# Some ML methods require scaling of the input feautures
# "Adaptive" scaling because it adapts to the distribution of the features

def scaling(dataframe, algorithm, cyclic_features=None, verbose=False, plot=False):

    scaling_method = None

    assert algorithm in ["linear_regression", "neural_network", "bart", "lstm", "random_forest", "xgboost"], \
        "Invalid algorithm specified."

    # Determine scaling method based on algorithm
    if algorithm in ["lstm", "neural_network"]:
        scaling_method = "minmax"
        if verbose: print(f"Algorithm '{algorithm}' selected: Using MinMax scaling.")
    elif algorithm == "linear_regression":
        scaling_method = "standard"
        if verbose: print(f"Algorithm '{algorithm}' selected: Using Standard scaling.")
    else:
        scaling_method = None
        if verbose: print(f"Algorithm '{algorithm}' selected: No scaling will be applied.")

    scaled_df = dataframe.copy()
    scaling_info = {}
    columns = [col for col in dataframe.columns]

    if verbose: print("Scaling in progress...\n")
    for col in columns:
        if verbose: print(f"Processing column: {col}")
        
        # Handle missing values
        if scaled_df[col].isnull().any():
            if verbose: print(f"Warning: Column '{col}' contains missing values. Imputing with median.")
            scaled_df[col].fillna(scaled_df[col].median(), inplace=True)

        # Cyclical encoding for specified features
        if cyclic_features and col in cyclic_features:
            max_value = scaled_df[col].max() + 1  # Assuming cyclic range [0, max_value - 1]
            if verbose: print(f"  Applying cyclical encoding (sine/cosine) for '{col}'.")
            scaled_df[f"{col}_sin"] = np.sin(2 * np.pi * scaled_df[col] / max_value)
            scaled_df[f"{col}_cos"] = np.cos(2 * np.pi * scaled_df[col] / max_value)
            scaling_info[col] = {"method": "cyclical_encoding"}
            scaled_df = scaled_df.drop(columns=col)
            continue

        # Skip scaling if no scaling method is selected
        if scaling_method is None:
            if verbose: print(f"No scaling applied for column: {col}")
            scaling_info[col] = {"method": "none"}
            continue

        # Determine the scaler
        if scaling_method == "minmax":
            scaler = MinMaxScaler()
            message = "Applying MinMaxScaler."
        elif scaling_method == "standard":
            scaler = StandardScaler()
            message = "Applying StandardScaler."
        elif scaling_method == "logminmax":
            scaled_df[col] = np.log1p(scaled_df[col] - scaled_df[col].min() + 1)
            scaler = MinMaxScaler()
            message = "Applying log transformation followed by MinMaxScaler."
        elif scaling_method == "individual":
            skewness = skew(scaled_df[col])
            _, p_value = normaltest(scaled_df[col])
            if verbose: print(f"  Skewness: {skewness:.2f}, Normality test p-value: {p_value:.4f}")
            if p_value > 0.05:  # Normally distributed
                scaler = StandardScaler()
                message = "Applying StandardScaler (data is approximately normal)."
            elif abs(skewness) > 1:  # Highly skewed
                scaled_df[col] = np.log1p(scaled_df[col] - scaled_df[col].min() + 1)
                scaler = MinMaxScaler()
                message = "Applying log transformation followed by MinMaxScaler (data is skewed)."
            else:  # Mildly skewed or uniform
                scaler = MinMaxScaler()
                message = "Applying MinMaxScaler (data is mildly skewed or uniform)."
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")

        if verbose: print(message)
        
        # Apply scaling
        scaled_values = scaler.fit_transform(scaled_df[col].values.reshape(-1, 1))
        scaled_df[col] = scaled_values.flatten()
        scaling_info[col] = {"method": type(scaler).__name__}

        if scaling_method == "individual":
            scaling_info[col].update({
                "skewness": skewness,
                "normality_p_value": p_value,
            })

        if plot: plot_scaling(dataframe, scaled_df, col)

    return scaled_df, scaling_info

    
    
def plot_scaling(original_df, scaled_df, col):
    """ Plot distributions before and after scaling, for each column """
    plt.figure(figsize=(12, 6))

    # Plot non-scaled distribution
    plt.subplot(1, 2, 1)
    plt.hist(original_df[col], bins=30, alpha=0.7, label='Non-Scaled')
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
  
#------------------------- Machine Learning model tuning: ------------------------------#
# in order to find the "best model" we need to tune the hyperparameters of the model
# this function uses a cross validation method to find the best combination of hyperparameters
# based to the training dataset

# Define the cross-validation strategy:
CV_scoring = 'neg_mean_absolute_error'   # e.g. 'neg_mean_squared_error', 'r2', 'neg_root_mean_squared_error. More available methods for regression evaluation (scoring): https://scikit-learn.org/1.5/modules/model_evaluation.html#scoring-parameter)
cv = 3  # Number of cross-validation folds


def model_tuning_CV(X_train, y_train, model, hyperparameters, cv = cv , scoring = CV_scoring, verbose=0):
    """
    Perform hyperparameter tuning using GridSearchCV.

    Parameters:
        X_train (array-like): Training feature matrix.
        y_train (array-like): Training target vector.
        model (object): Machine learning model to be tuned.
        hyperparameters (dict): Grid of hyperparameters to search.
        cv (int, optional): Number of cross-validation folds (default is 5).
        scoring (str, optional): Scoring metric for evaluation (default is 'accuracy').
        verbose (int, optional): Verbosity level of GridSearchCV (default is 0).

    Returns:
        dict: Best hyperparameters found by GridSearchCV.
        object: Best model fitted on the training data.
    """
    try:
        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=hyperparameters,
            cv=cv,
            n_jobs=-1,
            scoring=scoring,
            verbose=verbose
        )
        
        # Perform grid search on training data
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        print(f"Best Parameters: {best_params}")
        print(f"Best CV Score: {grid_search.best_score_:.2f}")
        
        # Get all hyperparameter results
        cv_results = grid_search.cv_results_

        return best_model, best_params, cv_results
    except Exception as e:
        print(f"Error during model tuning: {e}")
        return None, None



#------------------------- Machine Learning model evaluation: ------------------------------#
# evaluate the performance of the ML method based on the "scoring" (r2 or mean sqaured error) between the predicted dataset and the test dataset


def evaluate_model(model, X_test, y_test, scoring ='r2'):
    """
    Evaluate a trained model on the test data and compute metrics.

    Parameters:
        model (object): Trained machine learning model.
        X_test (array-like): Test feature matrix.
        y_test (array-like): Test target vector.
        scoring (str, optional): Scoring metric for evaluation (default is 'r2').

    Returns:
        dict: A dictionary containing predictions and evaluation metrics.
    """
    
    # Validate the scoring parameter
    if scoring not in {'r2', 'mse'}:
        raise ValueError("Invalid scoring metric. Allowed values are 'r2' and 'mse'.")
        
    try:
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Compute evaluation metrics based on the scoring method
        print(f"Test Metrics:")
        if scoring == 'r2':
            metrics = r2_score(y_test, y_pred)
            print(f"  R² Score: {metrics:.2f}")
        if scoring == 'mse':
            metrics = mean_squared_error(y_test, y_pred)
            print(f"  Mean Squared Error: {metrics:.2f}")
        
        return metrics
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return None

