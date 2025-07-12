from probatus.feature_elimination import EarlyStoppingShapRFECV
from skopt import BayesSearchCV
import lightgbm
import pandas as pd

def shap_bayes_feature_selection(X_train, y_train, X_test, cv_splitter, scoring_metric='recall',
                                   eval_metric='recall', early_stopping_rounds=30, step=0.2,
                                   random_state=26, verbose=2):
    """
    Perform SHAP-based feature elimination with Bayesian hyperparameter optimization using LightGBM.

    Parameters:
    - X_train: DataFrame, training features.
    - y_train: Series, training labels.
    - X_test: DataFrame, testing features (not used in elimination but can be returned).
    - cv_splitter: sklearn splitter object, e.g., StratifiedKFold.
    - scoring_metric: str, scoring metric used during cross-validation.
    - eval_metric: str, evaluation metric for early stopping.
    - early_stopping_rounds: int, number of rounds with no improvement to stop.
    - step: float, proportion of features to remove per iteration.
    - random_state: int, seed for reproducibility.
    - verbose: int, verbosity level.

    Returns:
    - reduced_X_train: DataFrame with selected features.
    - selected_features: list of selected feature names.
    - elimination_report: DataFrame containing elimination performance.
    """
    
    # Define base model
    base_model = lightgbm.LGBMClassifier(max_depth=5, class_weight="balanced")

    # Hyperparameter search space
    param_grid = {
        "n_estimators": [5, 7, 10],
        "num_leaves": [3, 5, 7, 10],
    }

    # Bayesian Optimization wrapper
    search = BayesSearchCV(
        estimator=base_model,
        search_spaces=param_grid,
        random_state=random_state
    )
    
    # SHAP-RFE with early stopping
    shap_elim = EarlyStoppingShapRFECV(
        model=search,
        step=step,
        cv=cv_splitter,
        scoring=scoring_metric,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=-1,
        verbose=verbose,
        random_state=random_state
    )

    # Run elimination
    report = shap_elim.fit_compute(X_train, y_train)

    # Get best feature set
    selected_features = shap_elim.get_reduced_features_set(num_features="best")
    final_features = shap_elim.get_reduced_features_set(num_features=len(selected_features))

    # Create reduced training set with selected features
    X_train_reduced = X_train[final_features]
    X_test_reduced = X_test[final_features]
   

    categorical_features_subset = X_train_reduced.select_dtypes(include=['category']).columns # Selecting the categorical features from the reduced training set
    categorical_indices_subset = [X_train_reduced.columns.get_loc(col) for col in categorical_features_subset] # Getting the indices of the categorical features for SMOTENC

    return X_train_reduced, X_test_reduced, categorical_indices_subset, selected_features
