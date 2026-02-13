from scripts.load_dataset import load_dataset
from scripts.split_and_preprocess import stratified_split, preprocess
from scripts.train_baseline_lr_model import train_logistic_regression
from scripts.feature_selection import shap_bayes_feature_selection
from scripts.train_xgb_with_bayesopt import train_xgb_with_bayes
from scripts.calibrate_model import calibrate_classifier
from scripts.evaluate_model import evaluate_and_plot
from scripts.explain_models import explain_model_with_shap_plots
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
    X_train_encoded, y_train_encoded, X_test_encoded, y_test_encoded, label_encoder, _ = preprocess(X_train, y_train, X_test, y_test)
    print("Training and testing sets preprocessed successfully.")

    # Create a StratifiedShuffleSplit object for cross-validation
    cv_splitter = StratifiedShuffleSplit(n_splits=10, random_state=26)

    # Feature selection before model training
    X_train_reduced, X_test_reduced, categorical_indices, selected_features =  shap_bayes_feature_selection(X_train_encoded, y_train_encoded, X_test_encoded, cv_splitter) 
    print("Feature selection completed successfully. Reduced features:", len(selected_features))

    # Train the baseline logistic regression model after feature selection
    lr_model = train_logistic_regression(X_train_reduced, y_train_encoded)
    print("Baseline logistic regression model trained successfully.")

    # Evaluate the baseline model
    evaluate_and_plot(lr_model, X_test_reduced, y_test_encoded, label_encoder, "./results/LR", "LR",
                      model_name="Logistic Regression", raw_model=None)
    print("Baseline logistic regression model evaluated successfully.")

    # Explain the baseline model using SHAP
    explain_model_with_shap_plots(lr_model, X_train_reduced, X_test_reduced, "./results/LR",
                                  classifier_step_name="logisticregression", sample_idx=2)
    print("SHAP summary plot for baseline logistic regression model generated successfully.")
    
    print("\n" + "="*60)
    print("TRAINING BLACK BOX MODELS WITH BAYESIAN OPTIMIZATION")
    print("="*60)
    
    # Train XGBoost model with reduced features and Bayesian optimization
    print("\nTraining XGBoost model with Bayesian optimization...")
    xgb_model = train_xgb_with_bayes(X_train_reduced, y_train_encoded, categorical_indices, cv_splitter)
    print("XGBoost model trained with Bayesian optimization successfully.")

    # Train Random Forest model with reduced features and Bayesian optimization
    print("\nTraining Random Forest model with Bayesian optimization...")
    rf_model = train_rf_with_bayes(X_train_reduced, y_train_encoded, categorical_indices, cv_splitter)
    print("Random Forest model trained with Bayesian optimization successfully.")
    
    print("\n" + "="*60)
    print("STARTING MODEL CALIBRATION AND EVALUATION")
    print("="*60)

    # Calibrate the XGBoost model
    calibrated_xgb_model = calibrate_classifier(xgb_model, X_train_reduced, y_train_encoded, cv_splitter)
    print("XGBoost model calibrated successfully.")

    #  Evaluate the calibrated XGBoost model
    evaluate_and_plot(calibrated_xgb_model, X_test_reduced, y_test_encoded, label_encoder, "./results/XGB", "XGB",
                      model_name="XGBoost", raw_model=xgb_model)
    print("Calibrated XGBoost model evaluated successfully.")
    
    # Explain the calibrated XGBoost model using SHAP
    explain_model_with_shap_plots(calibrated_xgb_model, X_train_reduced, X_test_reduced, "./results/XGB", classifier_step_name="classifier",
                                  sample_idx=2)
    
    print("SHAP summary plot for calibrated XGBoost model generated successfully.")

    # Calibrate the Random Forest model
    calibrated_rf_model = calibrate_classifier(rf_model, X_train_reduced, y_train_encoded, cv_splitter)
    print("Random Forest model calibrated successfully.")

    # Evaluate the calibrated Random Forest model
    evaluate_and_plot(calibrated_rf_model, X_test_reduced, y_test_encoded, label_encoder, "./results/RF", "RF",
                      model_name="Random Forest", raw_model=rf_model)
    print("Calibrated Random Forest model evaluated successfully.")

    # Explain the calibrated Random Forest model using SHAP
    explain_model_with_shap_plots(calibrated_rf_model, X_train_reduced, X_test_reduced, "./results/RF", classifier_step_name="classifier", 
                                  sample_idx=2)
    print("SHAP summary plot for calibrated Random Forest model generated successfully.")

    print("\n" + "="*60)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("All model evaluations and SHAP plots saved to respective result folders.")

if __name__ == "__main__":
    dataset_path = input("Enter the path to the dataset: ")
    main(dataset_path)
