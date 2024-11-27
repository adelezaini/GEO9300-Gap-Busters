
# BART model setup

from bartpy.sklearnmodel import SklearnModel

# BART model (assuming this is copy-pasted into the overall workflow script

    elif algorithm == 'bart':
    
        # BART REGRESSOR
        
        # Apply scaling to input features
        X_scaled, _ = scaling(X, algorithm, cyclic_features = ['hour','doy'])
        
        # Split training and testing datasets
        X_train, y_train, X_test, y_test = split_train_test_dataset(X_scaled, y)

        bart_model = BART(n_trees=50, n_burn=1000, n_samples=1000)
        # Valid parameters are: ['alpha', 'beta', 'initializer', 'n_burn', 'n_chains', 'n_jobs', 'n_samples', 'n_trees', 'sigma_a', 'sigma_b', 'store_acceptance_trace', 'store_in_sample_predictions', 'thin', 'tree_sampler'].

        param_grid_bart = {
            'n_trees': [50, 100, 200],
            'alpha': [0.95, 0.99],
            'beta': [1.0, 2.0],
        }

        BART_best_model, BART_best_params, cv_results = model_tuning_CV(X_train, y_train, bart_model, param_grid_bart, cv, CV_scoring)
        BART_metrics = evaluate_model(BART_best_model, X_test, y_test, scoring_metrics)
        metric = BART_metrics # for the general saving code at the end of the script

        # Generate predictions
        y_pred = BART_best_model.predict(X_test)

        print("\n=== BART RESULTS ===")
        print(f"Best Parameters: {BART_best_params}")
        print(f"Test Metrics: {BART_metrics}")