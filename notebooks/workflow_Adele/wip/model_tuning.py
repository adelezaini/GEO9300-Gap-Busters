# Cross-validation method to tune the hyperparameters of the models


from sklearn.model_selection import GridSearchCV


def model_tuning_CV(X_train, y_train, X_test, Y_test, model, hyperparameters, cv, scoring):
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
