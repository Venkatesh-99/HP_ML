# Script written on 2025-06-13

from scripts.load_dataset import load_dataset
from scripts.split_and_preprocess import stratified_split, preprocess
from scripts.train_baseline_lr_model import train_logistic_regression
# from scripts.evaluate_baseline_lr_model import evaluate_model
# from scripts.explain_baseline_lr_model import shap_summary_plot
from scripts.feature_selection import shap_bayes_feature_selection
from scripts.train_xgb_with_bayesopt import train_xgb_with_bayes
from scripts.calibrate_model import calibrate_classifier
from scripts.evaluate_model import evaluate_and_plot
from scripts.explain_black_box_models import explain_model_with_shap_plots
from scripts.train_rf_with_bayesopt import train_rf_with_bayes
from scripts.train_svm_with_bayesopt import train_svm_with_bayes

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def extract_cv_results(model, model_name):
    """
    Extract cross-validation results from BayesSearchCV model
    """
    cv_results = model.cv_results_
    best_score = model.best_score_
    best_std = cv_results['std_test_score'][model.best_index_]
    best_params = model.best_params_
    
    # Get all CV scores for the best model
    # Extract scores for all folds of the best parameter combination
    all_scores = []
    for i in range(10):  # assuming 10 folds
        score_key = f'split{i}_test_score'
        if score_key in cv_results:
            all_scores.append(cv_results[score_key][model.best_index_])
    
    return {
        'model': model_name,
        'mean_recall': best_score,
        'std_recall': best_std,
        'best_params': best_params,
        'cv_scores': all_scores
    }

def create_cv_results_table_and_plots(cv_results_list):
    """
    Create combined violin plot with embedded table from CV results
    """
    # Create results directory if it doesn't exist
    os.makedirs("./results", exist_ok=True)
    
    # Create summary table
    summary_data = []
    all_scores_data = []
    
    for result in cv_results_list:
        summary_data.append({
            'Model': result['model'],
            'Mean Recall': f"{result['mean_recall']:.3f}",
            'Std Recall': f"{result['std_recall']:.3f}",
            'Mean ± SD': f"{result['mean_recall']:.3f} ± {result['std_recall']:.3f}"
        })
        
        # Prepare data for violin plot
        for score in result['cv_scores']:
            all_scores_data.append({
                'Model': result['model'],
                'Recall': score
            })
    
    # Create summary table
    summary_df = pd.DataFrame(summary_data)
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)
    
    # Save table to CSV
    summary_df.to_csv('./results/cv_results_summary.csv', index=False)
    print(f"Summary table saved to: ./results/cv_results_summary.csv")
    
    # Create combined violin plot with table
    scores_df = pd.DataFrame(all_scores_data)
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 10))
    
    # Create violin plot (takes up top 70% of figure)
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    
    # Violin plot
    violin_parts = ax1.violinplot([scores_df[scores_df['Model'] == model]['Recall'].values 
                                  for model in summary_df['Model']], 
                                 positions=range(len(summary_df)), 
                                 showmeans=True, showmedians=True)
    
    # Customize violin plot colors
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
    for i, pc in enumerate(violin_parts['bodies']):
        if i < len(colors):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
    
    # Add mean points as diamonds
    for i, result in enumerate(cv_results_list):
        ax1.scatter(i, result['mean_recall'], color='red', s=120, zorder=5, marker='D', 
                   label='Mean' if i == 0 else "", edgecolors='darkred', linewidth=1.5)
    
    ax1.set_xticks(range(len(summary_df)))
    ax1.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
    ax1.set_ylabel('Cross-Validation Recall', fontsize=12, fontweight='bold')
    ax1.set_title('Cross-Validation Recall Score Distributions', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0.6, 1.0)
    
    # Add legend
    ax1.legend(loc='upper left')
    
    # Create table (takes up bottom 30% of figure)
    ax2 = plt.subplot2grid((4, 1), (3, 0))
    ax2.axis('tight')
    ax2.axis('off')
    
    # Prepare table data for display
    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([row['Model'], row['Mean ± SD']])
    
    # Create table
    table = ax2.table(cellText=table_data,
                     colLabels=['Model', 'Cross-Validation Recall (Mean ± SD)'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.4, 0.6])
    
    # Customize table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Color the header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternate rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    # Add border to table
    for i in range(len(table_data) + 1):
        for j in range(len(table_data[0])):
            table[(i, j)].set_edgecolor('black')
            table[(i, j)].set_linewidth(1)
    
    plt.tight_layout()
    plt.savefig('./results/cv_recall_violin_with_table.png', dpi=1200, bbox_inches='tight')
    plt.show()
    
    print(f"Combined visualization saved to: ./results/cv_recall_violin_with_table.png")
    
    return summary_df, scores_df

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

    # Train the baseline logistic regression model
    lr_model = train_logistic_regression(X_train_encoded, y_train_encoded)
    print("Baseline logistic regression model trained successfully.")

    # Evaluate the baseline model
    evaluate_and_plot(lr_model, X_test_encoded, y_test_encoded, label_encoder, "./results/2025-08-29_LR", "LR",
                      model_name="Logistic Regression", raw_model=None)
    print("Baseline logistic regression model evaluated successfully.")

    # Explain the baseline model using SHAP
    explain_model_with_shap_plots(lr_model, X_train_encoded, X_test_encoded, "./results/2025-08-29_LR",
                                  classifier_step_name="logisticregression", sample_idx=3, class_index=None)
    print("SHAP summary plot for baseline logistic regression model generated successfully.")

    # Create a StratifiedShuffleSplit object for cross-validation
    cv_splitter = StratifiedShuffleSplit(n_splits=10, random_state=26)

    # Feature selection for black box models
    X_train_reduced, X_test_reduced, categorical_indices, selected_features =  shap_bayes_feature_selection(X_train_encoded, y_train_encoded, X_test_encoded, cv_splitter) 
    print("Feature selection completed successfully. Reduced features:", len(selected_features))
    
    print("\n" + "="*60)
    print("STARTING CROSS-VALIDATION ANALYSIS")
    print("="*60)
    
    # Initialize list to store CV results
    cv_results_list = []
    
    # For Logistic Regression (no hyperparameter tuning, so perform manual CV)
    print("Extracting CV results for Logistic Regression...")
    lr_cv_scores = cross_val_score(lr_model, X_train_encoded, y_train_encoded, 
                                   cv=cv_splitter, scoring='recall')
    cv_results_list.append({
        'model': 'Logistic Regression',
        'mean_recall': lr_cv_scores.mean(),
        'std_recall': lr_cv_scores.std(),
        'best_params': 'No hyperparameter tuning',
        'cv_scores': lr_cv_scores.tolist()
    })
    print(f"LR CV Results - Mean: {lr_cv_scores.mean():.3f} ± {lr_cv_scores.std():.3f}")
    
    # Train XGBoost model with reduced features and Bayesian optimization
    print("\nTraining XGBoost model with Bayesian optimization...")
    xgb_model = train_xgb_with_bayes(X_train_reduced, y_train_encoded, categorical_indices, cv_splitter)
    xgb_cv_result = extract_cv_results(xgb_model, 'XGBoost')
    cv_results_list.append(xgb_cv_result)
    print(f"XGBoost CV Results - Mean: {xgb_cv_result['mean_recall']:.3f} ± {xgb_cv_result['std_recall']:.3f}")
    print("XGBoost model trained with Bayesian optimization successfully.")

    # Train Random Forest model with reduced features and Bayesian optimization
    print("\nTraining Random Forest model with Bayesian optimization...")
    rf_model = train_rf_with_bayes(X_train_reduced, y_train_encoded, categorical_indices, cv_splitter)
    rf_cv_result = extract_cv_results(rf_model, 'Random Forest')
    cv_results_list.append(rf_cv_result)
    print(f"RF CV Results - Mean: {rf_cv_result['mean_recall']:.3f} ± {rf_cv_result['std_recall']:.3f}")
    print("Random Forest model trained with Bayesian optimization successfully.")

    # Train SVM model with reduced features and Bayesian optimization
    print("\nTraining SVM model with Bayesian optimization...")
    svm_model = train_svm_with_bayes(X_train_reduced, y_train_encoded, categorical_indices, cv_splitter)
    svm_cv_result = extract_cv_results(svm_model, 'SVM')
    cv_results_list.append(svm_cv_result)
    print(f"SVM CV Results - Mean: {svm_cv_result['mean_recall']:.3f} ± {svm_cv_result['std_recall']:.3f}")
    print("SVM model trained with Bayesian optimization successfully.")
    
    # Create and save CV results table and plots
    print("\nGenerating CV results summary and visualizations...")
    summary_df, scores_df = create_cv_results_table_and_plots(cv_results_list)
    
    print("\n" + "="*60)
    print("STARTING MODEL CALIBRATION AND EVALUATION")
    print("="*60)

    # Calibrate the XGBoost model
    calibrated_xgb_model = calibrate_classifier(xgb_model, X_train_reduced, y_train_encoded, cv_splitter)
    print("XGBoost model calibrated successfully.")

    #  Evaluate the calibrated XGBoost model
    evaluate_and_plot(calibrated_xgb_model, X_test_reduced, y_test_encoded, label_encoder, "./results/2025-08-29_XGB", "XGB",
                      model_name="XGBoost", raw_model=xgb_model)
    print("Calibrated XGBoost model evaluated successfully.")
    
    # Explain the calibrated XGBoost model using SHAP
    explain_model_with_shap_plots(calibrated_xgb_model, X_train_reduced, X_test_reduced, "./results/2025-08-29_XGB", classifier_step_name="classifier",
                                  sample_idx=3, class_index=None)
    
    print("SHAP summary plot for calibrated XGBoost model generated successfully.")

    # Calibrate the Random Forest model
    calibrated_rf_model = calibrate_classifier(rf_model, X_train_reduced, y_train_encoded, cv_splitter)
    print("Random Forest model calibrated successfully.")

    # Evaluate the calibrated Random Forest model
    evaluate_and_plot(calibrated_rf_model, X_test_reduced, y_test_encoded, label_encoder, "./results/2025-08-29_RF", "RF",
                      model_name="Random Forest", raw_model=rf_model)
    print("Calibrated Random Forest model evaluated successfully.")

    # Explain the calibrated Random Forest model using SHAP
    explain_model_with_shap_plots(calibrated_rf_model, X_train_reduced, X_test_reduced, "./results/2025-08-29_RF", classifier_step_name="classifier", 
                                  sample_idx=3, class_index=0)
    print("SHAP summary plot for calibrated Random Forest model generated successfully.")

    # Calibrate the SVM model
    calibrated_svm_model = calibrate_classifier(svm_model, X_train_reduced, y_train_encoded, cv_splitter)
    print("SVM model calibrated successfully.")   

    # Evaluate the calibrated SVM model
    evaluate_and_plot(calibrated_svm_model, X_test_reduced, y_test_encoded, label_encoder, "./results/2025-08-29_SVM", "SVM",
                      model_name="Support Vector Machine", raw_model=svm_model)
    print("Calibrated SVM model evaluated successfully.")

    # Explain the calibrated SVM model using SHAP
    explain_model_with_shap_plots(calibrated_svm_model, X_train_reduced, X_test_reduced, "./results/2025-08-29_SVM", classifier_step_name="classifier", 
                                  sample_idx=3, class_index=0)
    print("SHAP summary plot for calibrated SVM model generated successfully.")

    print("\n" + "="*60)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"CV Results Summary saved to: ./results/cv_results_summary.csv")
    print(f"CV Visualization with table saved to: ./results/cv_recall_violin_with_table.png")
    print("All model evaluations and SHAP plots saved to respective result folders.")

if __name__ == "__main__":
    dataset_path = input("Enter the path to the dataset: ")
    main(dataset_path)