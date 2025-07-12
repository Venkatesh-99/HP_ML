# Script written on 2025-06-13

from scripts.load_dataset import load_dataset
from scripts.split_and_preprocess import stratified_split, preprocess
from scripts.train_baseline_lr_model import train_logistic_regression
from scripts.evaluate_baseline_lr_model import evaluate_model
from scripts.explain_baseline_lr_model import shap_summary_plot
from scripts.feature_selection import shap_bayes_feature_selection
from scripts.train_xgb_with_bayesopt import train_xgb_with_bayes
from scripts.calibrate_model import calibrate_classifier
from scripts.evaluate_model import evaluate_and_plot
from scripts.explain_black_box_models import plot_shap_summary
from scripts.train_rf_with_bayesopt import train_rf_with_bayes

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedShuffleSplit

def main(dataset_path):
    """
    Main function to execute the entire workflow of loading, cleaning, splitting,
    preprocessing, training, evaluating, and explaining models.
    Parameters:
    - dataset_path: str, path to the dataset file.
    """

    # Load the dataset
    df = load_dataset(dataset_path, sheet_name="Sheet1")

    # Clean the dataset
    # df = clean_data(df)
    print("Dataset loaded and cleaned successfully.")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = stratified_split(df)
    print("Dataset split into training and testing sets successfully.")

    # Preprocess the training and testing sets
    X_train_encoded, y_train_encoded, X_test_encoded, y_test_encoded, label_encoder, one_hot_encoder = preprocess(X_train, y_train, X_test, y_test)
    print("Training and testing sets preprocessed successfully.")

    # Train the baseline logistic regression model
    lr_model = train_logistic_regression(X_train_encoded, y_train_encoded)
    print("Baseline logistic regression model trained successfully.")

    # Evaluate the baseline model
    evaluate_model(lr_model, X_test_encoded, y_test_encoded, label_encoder, "results")
    print("Baseline logistic regression model evaluated successfully.")

    # Explain the baseline model using SHAP
    shap_summary_plot(lr_model, X_test_encoded, "results")
    print("SHAP summary plot for basline logistic regression model generated successfully.")

    # Create a StratifiedShuffleSplit object for cross-validation
    cv_splitter = StratifiedShuffleSplit(n_splits=10, random_state=26)

    # Feature selection for black box models
    X_train_reduced, X_test_reduced, categorical_indices, selected_features =  shap_bayes_feature_selection(X_train_encoded, y_train_encoded, X_test_encoded, cv_splitter) 
    # print("Feature selection completed successfully. Reduced features:", selected_features)
    # Train XGBoost model with reduced features and Bayesian optimization
    xgb_model = train_xgb_with_bayes(X_train_reduced, y_train_encoded, categorical_indices, cv_splitter)
    print("XGBoost model trained with Bayesian optimization successfully.")

    # Calibrate the XGBoost model
    calibrated_xgb_model = calibrate_classifier(xgb_model, X_train_reduced, y_train_encoded, cv_splitter)
    print("XGBoost model calibrated successfully.")

    #  Evaluate the calibrated XGBoost model
    evaluate_and_plot(calibrated_xgb_model, X_test_reduced, y_test_encoded, label_encoder, "XGB", "./results/")
    print("Calibrated XGBoost model evaluated successfully.")
    
    # Explain the calibrated XGBoost model using SHAP
    plot_shap_summary(calibrated_xgb_model, X_test_reduced, "./results/figures/XGB_calibrated_shap_summary.png", class_index=0)
    print("SHAP summary plot for calibrated XGBoost model generated successfully.")

    # Train Random Forest model with reduced features and Bayesian optimization
    rf_model = train_rf_with_bayes(X_train_reduced, y_train_encoded, categorical_indices, cv_splitter)
    print("Random Forest model trained with Bayesian optimization successfully.")

    # Calibrate the Random Forest model
    calibrated_rf_model = calibrate_classifier(rf_model, X_train_reduced, y_train_encoded, cv_splitter)
    print("Random Forest model calibrated successfully.")

    # Evaluate the calibrated Random Forest model
    evaluate_and_plot(calibrated_rf_model, X_test_reduced, y_test_encoded, label_encoder, "RF", "./results/")
    print("Calibrated Random Forest model evaluated successfully.")

    # Explain the calibrated Random Forest model using SHAP
    plot_shap_summary(calibrated_rf_model, X_test_reduced, "./results/figures/RF_calibrated_shap_summary.png", class_index=1)
    print("SHAP summary plot for calibrated Random Forest model generated successfully.")

if __name__ == "__main__":
    dataset_path = input("Enter the path to the dataset: ")
    main(dataset_path)