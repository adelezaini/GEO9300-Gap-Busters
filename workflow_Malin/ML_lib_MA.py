import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, normaltest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


#------------------------- Adaptive scaling: ------------------------------#
# Some ML methods require scaling of the input feautures
# "Adaptive" scaling because it adapts to the distribution of the features

def adaptive_scaling(dataframe, scaling_method="individual", cyclic_features = None, verbose = False, plot = False):
    """
    Scales the features in the dataframe based on the specified or adaptive scaling method.
    
    Parameters:
        dataframe (pd.DataFrame): Input dataset.
        scaling_method (str): Scaling method to use. Options are:
            - "individual": Decide scaling per column based on distribution - adaptive scaling ON
            - "minmax": Use MinMaxScaler for all features.
            - "standard": Use StandardScaler for all features.
            - "logminmax": Apply log transformation followed by MinMaxScaler for all features.
        cyclic_features (list): List of column names that represent cyclical features (e.g., "hour", "doy").
        verbose (bool): If True, print details during processing.
        plot (bool): If True, plot distributions before and after scaling.
    Returns:
        scaled_df (pd.DataFrame): Scaled dataset with the same columns as the input.
        scaling_info (dict): Information about the scaling method used for each column.
    """
    assert scaling_method in ["individual", "minmax", "standard", "logminmax"], \
        "Invalid scaling_method. Choose from 'individual', 'minmax', 'standard', 'logminmax'."
    
    scaled_df = dataframe.copy()
    scaling_info = {}
    columns = [col for col in dataframe.columns]

    if verbose: print("Scaling in progress..\n")
    for col in columns:
        
        if verbose: print(f"Processing column: {col}")
        
        # Check for missing values
        if scaled_df[col].isnull().any():
            if verbose: print(f"Warning: Column '{col}' contains missing values. Imputing with median.")
            scaled_df[col].fillna(scaled_df[col].median(), inplace=True)
            
        # Cyclical encoding for specified features:
          # Add two columns col_sin and col_cos because they capture two different parts of the periodic behaviour
        if cyclic_features and col in cyclic_features:
            max_value = scaled_df[col].max() + 1  # Assuming cyclic range [0, max_value - 1] #ex: hour 24, doy 365
            if verbose: print(f"  Applying cyclical encoding (sine/cosine) for '{col}'.")
            scaled_df[f"{col}_sin"] = np.sin(2 * np.pi * scaled_df[col] / max_value)
            scaled_df[f"{col}_cos"] = np.cos(2 * np.pi * scaled_df[col] / max_value)
            scaling_info[col] = {"method": "cyclical_encoding"}
            scaled_df = scaled_df.drop(columns = col)
            continue
        else:

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
        print(X_train)
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
        
      #  print(f"Best Parameters: {best_params}")
      #  print(f"Best CV Score: {grid_search.best_score_:.2f}")
        
        # Get all hyperparameter results
        cv_results = grid_search.cv_results_

        return best_model, best_params, cv_results
    except Exception as e:
        print(f"Error during model tuning: {e}")
        return None, None



#------------------------- Machine Learning model evaluation: ------------------------------#
# evaluate the performance of the ML method based on the "scoring" (r2 or mean sqaured error) between the predicted dataset and the test dataset


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_model(model, X_test, y_test, scoring=['r2']):
    """
    Evaluate a trained model on the test data and compute metrics.

    Parameters:
        model (object): Trained machine learning model.
        X_test (array-like): Test feature matrix.
        y_test (array-like): Test target vector.
        scoring (list, optional): List of scoring metrics for evaluation (default is ['r2']).

    Returns:
        dict: A dictionary containing predictions and evaluation metrics.
    """
    
    # Validate the scoring parameter
    valid_metrics = {'r2', 'mse', 'mae'}
    if not all(metric in valid_metrics for metric in scoring):
        raise ValueError(f"Invalid scoring metric. Allowed values are {valid_metrics}.")
        
    try:
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Compute evaluation metrics based on the scoring method
        metrics = {}
        print(f"Test Metrics:")
        for metric in scoring:
            if metric == 'r2':
                metrics['r2'] = r2_score(y_test, y_pred)
                print(f"  R² Score: {metrics['r2']:.2f}")
            elif metric == 'mse':
                metrics['mse'] = mean_squared_error(y_test, y_pred)
                print(f"  Mean Squared Error: {metrics['mse']:.2f}")
            elif metric == 'mae':
                metrics['mae'] = mean_absolute_error(y_test, y_pred)
                print(f"  Mean Absolute Error: {metrics['mae']:.2f}")
        
        return metrics
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return None

#------------------------- Save cross-validation and evaluation metrics results to CSV: ------------------------------#
# save to csv:

def save_results_to_csv(cv_results, metrics, algorithm, gaps_data_file, CV_scoring, save_to_csv=True):
    """
    Save cross-validation results to a CSV file.

    Parameters:
    cv_results (dict): Cross-validation results.
    algorithm (str): Name of the algorithm.
    gaps_data_file (str): Name of the gaps data file.
    CV_scoring (str): Scoring metric used for cross-validation.
    save_to_csv (bool): Flag to save the results to CSV. Default is True.
    """
    if save_to_csv:
        # CV results:
        cv_results_df = pd.DataFrame(cv_results)
        cv_results_df.to_csv(f'../results/{algorithm}_{gaps_data_file}_{CV_scoring}_cv_results.csv', index=False)

        # Save evaluation metrics
        metrics_summary = {
            'Algorithm': [algorithm],
            'Gaps Data File': [gaps_data_file],
            'CV Scoring': [CV_scoring]
        }
        metrics_summary.update(metrics)
        metrics_df = pd.DataFrame(metrics_summary)
        metrics_df.to_csv(f'../results/{algorithm}_{gaps_data_file}_{CV_scoring}_metrics_summary.csv', index=False)

        # save y_predic and y_test in the same dataframe to compare them
        y_pred = pd.DataFrame(y_pred)
        y_test = pd.DataFrame(y_test)
        y_pred.to_csv(f'../results/{algorithm}_{gaps_data_file}_{CV_scoring}_y_pred.csv', index=False)
        y_test.to_csv(f'../results/{algorithm}_{gaps_data_file}_{CV_scoring}_y_test.csv', index=False)
        

