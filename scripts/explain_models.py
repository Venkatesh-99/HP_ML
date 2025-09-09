import os
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def explain_model_with_shap_plots(
    pipeline, X_train, X_test, save_dir, classifier_step_name="classifier", 
    sample_idx=0, only_explain_gc=True
):
    """
    Generates SHAP global summary plot and SHAP waterfall plot for a specific sample.
    Parameters:
    - pipeline: Trained sklearn Pipeline or model.
    - X_train: DataFrame, training features.
    - X_test: DataFrame, test features.
    - save_dir: str, directory to save the plots.
    - classifier_step_name: str, name of the classifier step in the pipeline.
    - sample_idx: int, index of the test sample to explain.
    - only_explain_gc: bool, if True, only explain if the sample is classified as gastric cancer.
    Returns:
    - None (plots are saved to disk).
    """
    # Set larger font sizes for all plots
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)

    # Convert categorical variables to numeric
    X_train_num = X_train.copy()
    X_test_num = X_test.copy()
    
    for col in X_train_num.select_dtypes(include="category").columns:
        X_train_num[col] = X_train_num[col].cat.codes
        X_test_num[col] = X_test_num[col].cat.codes

    X_train_num = X_train_num.astype(float)
    X_test_num = X_test_num.astype(float)
    
    # Handle problematic values
    X_train_num = X_train_num.replace([np.inf, -np.inf], np.nan)
    X_test_num = X_test_num.replace([np.inf, -np.inf], np.nan)
    X_train_num = X_train_num.fillna(X_train_num.median())
    X_test_num = X_test_num.fillna(X_train_num.median())

    # Auto-detect classifier step
    if hasattr(pipeline, "named_steps"):  
        model_step_names = list(pipeline.named_steps.keys())
        classifier_step_name = model_step_names[-1]
        model = pipeline.named_steps[classifier_step_name]
    elif hasattr(pipeline, "estimator") and hasattr(pipeline.estimator, "named_steps"):  
        model_step_names = list(pipeline.estimator.named_steps.keys())
        classifier_step_name = model_step_names[-1]
        model = pipeline.estimator.named_steps[classifier_step_name]
    else:
        raise ValueError("Could not find classifier step in pipeline.")

    print(f"Using classifier: {classifier_step_name} ({type(model).__name__})")

    # Select appropriate SHAP explainer
    if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier)):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, LogisticRegression):
        background_sample = shap.sample(X_train_num, 100)
        explainer = shap.LinearExplainer(model, background_sample)
    else:
        background_sample = shap.sample(X_train_num, 50)
        explainer = shap.PermutationExplainer(model.predict, background_sample)

    # Calculate SHAP values
    print("Computing SHAP values...")
    shap_values = explainer(X_test_num)

    if sample_idx >= len(X_test_num):
        sample_idx = 0
    
    sample_pred = pipeline.predict(X_test_num.iloc[[sample_idx]])[0]
    sample_probs = pipeline.predict_proba(X_test_num.iloc[[sample_idx]])[0]

    predicted_class = "Gastric cancer" if sample_pred == 0 else "Non-gastric cancer"
    print(f"Sample {sample_idx} predicted as: {predicted_class}")

    if only_explain_gc and sample_pred != 0:
        print(f"Skipping explanation: Sample not classified as gastric cancer.")
        return

    if hasattr(shap_values, 'values'):
        if shap_values.values.ndim == 3:
            values_for_bar = shap_values.values[:, :, 0]
            waterfall_values = shap_values.values[sample_idx, :, 0]
            expected_value = shap_values.base_values[sample_idx, 0]
        else:
            values_for_bar = shap_values.values
            waterfall_values = shap_values.values[sample_idx]
            expected_value = shap_values.base_values[sample_idx]
        feature_values = shap_values.data
    else:
        values_for_bar = shap_values
        waterfall_values = shap_values[sample_idx]
        expected_value = explainer.expected_value
        feature_values = X_test_num.values

    # Create SHAP explanation object for bar plot
    shap_explanation_for_bar = shap.Explanation(
        values=values_for_bar,
        base_values=expected_value,
        data=feature_values,
        feature_names=X_test_num.columns.tolist()
    )

    # Create explanation object for waterfall plot
    explanation_obj = shap.Explanation(
        values=waterfall_values,
        base_values=expected_value,
        data=X_test_num.iloc[sample_idx].values,
        feature_names=X_test_num.columns.tolist()
    )

    # Create combined plot with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 40), gridspec_kw={'hspace': 0.8})
    
    # Global SHAP bar plot
    ax1.text(-0.02, 1.25, 'A', transform=ax1.transAxes, fontsize=20, fontweight='bold', va='top')
    shap.plots.bar(shap_explanation_for_bar, ax=ax1, show=False, max_display=10)
    ax1.set_title('Global Feature Importance', fontsize=16, fontweight='bold', pad=20)
    
    # Individual waterfall plot 
    ax2.text(-0.02, 1.4, 'B', transform=ax2.transAxes, fontsize=20, fontweight='bold', va='top')
    plt.sca(ax2)  
    shap.waterfall_plot(explanation_obj, show=False)
    ax2.set_title(f'Individual Prediction Explanation - Sample {sample_idx}', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout(pad=2.0)
    combined_path = os.path.join(save_dir, "figures", f"shap_combined_plots_sample_{sample_idx}.png")
    plt.savefig(combined_path, dpi=1200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Combined SHAP plots saved to: {combined_path}")