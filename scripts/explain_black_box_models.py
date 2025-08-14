import os
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def explain_model_with_shap_plots(
    pipeline, X_train, X_test, save_dir, classifier_step_name="classifier", 
    sample_idx=0, class_index=None
):
    """
    Generates SHAP global summary plot and SHAP waterfall plot for a specific sample.

    Parameters:
    - pipeline: sklearn Pipeline containing the classifier as `classifier_step_name`.
    - X_train, X_test: pandas DataFrames (categorical columns should be category dtype).
    - save_dir: directory to save figures.
    - classifier_step_name: str, name of the classifier step in the pipeline.
    - sample_idx: int, index of the sample from X_test to explain with SHAP waterfall plot.
    - class_index: int, for multiclass models, which class to explain (0 for Gastric cancer).
    """

    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)

    # Ensure categorical variables are numeric
    X_train_num = X_train.copy()
    X_test_num = X_test.copy()
    for col in X_train_num.select_dtypes(include="category").columns:
        X_train_num[col] = X_train_num[col].cat.codes
        X_test_num[col] = X_test_num[col].cat.codes

    # --- Auto-detect classifier step ---
    if hasattr(pipeline, "named_steps"):  
        # Standard sklearn Pipeline
        model_step_names = list(pipeline.named_steps.keys())
        classifier_step_name = model_step_names[-1]  # usually last step
        model = pipeline.named_steps[classifier_step_name]
    elif hasattr(pipeline, "estimator") and hasattr(pipeline.estimator, "named_steps"):  
        # Wrapped pipeline (e.g., CalibratedClassifierCV, GridSearchCV)
        model_step_names = list(pipeline.estimator.named_steps.keys())
        classifier_step_name = model_step_names[-1]
        model = pipeline.estimator.named_steps[classifier_step_name]
    else:
        raise ValueError("Could not find classifier step in pipeline.")

    print(f"Detected classifier step: {classifier_step_name} ({type(model).__name__})")

    # --- Select SHAP Explainer type based on model ---
    if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier)):
        print("Using TreeExplainer for tree-based model.")
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, LogisticRegression):
        print("Using LinearExplainer for logistic regression.")
        explainer = shap.LinearExplainer(model, X_train_num)
    else:
        print("Using generic Explainer.")
        explainer = shap.Explainer(model, X_train_num)

    shap_values = explainer.shap_values(X_test_num)

    # --- Handle multi-class or 3D SHAP outputs ---
    if isinstance(shap_values, list):  # Multi-class case
        if class_index is None:
            raise ValueError(f"`class_index` required for multi-class models (found {len(shap_values)} classes).")
        values_to_plot = shap_values[class_index]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:  # 3D case
        if class_index is None:
            raise ValueError(f"`class_index` required for 3D SHAP values.")
        values_to_plot = shap_values[:, :, class_index]
    else:  # Single output
        values_to_plot = shap_values

    # --- Plot SHAP summary ---
    shap.summary_plot(
        values_to_plot, X_test_num,
        feature_names=X_test_num.columns.tolist(),
        plot_size=(20, 10),
        plot_type="bar",
        show=False
    )
    shap_path = os.path.join(save_dir, "figures", "shap_summary_plot.png")
    plt.savefig(shap_path, dpi=1200, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to: {shap_path}")

    # --- SHAP Waterfall Plot for individual sample ---
    print(f"Generating SHAP waterfall plot for sample {sample_idx}...")
    
    # Get actual prediction for this sample
    sample_pred = pipeline.predict(X_test_num.iloc[[sample_idx]])[0]
    sample_probs = pipeline.predict_proba(X_test_num.iloc[[sample_idx]])[0]
    
    print(f"Sample {sample_idx} - Predicted class: {sample_pred}")
    print(f"Probabilities: [Class 0 (Gastric): {sample_probs[0]:.3f}, Class 1 (Non-gastric): {sample_probs[1]:.3f}]")
    
    # For waterfall plot, get SHAP values for the specific sample and class
    if isinstance(shap_values, list):  # Multi-class case
        if class_index is None:
            class_index = sample_pred  # Default to explaining the predicted class
        waterfall_values = shap_values[class_index][sample_idx]
        expected_value = explainer.expected_value[class_index]
    else:  # Single output or need to select from 3D
        if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            if class_index is None:
                class_index = sample_pred
            waterfall_values = shap_values[sample_idx, :, class_index]
            expected_value = explainer.expected_value[class_index]
        else:
            waterfall_values = shap_values[sample_idx]
            expected_value = explainer.expected_value
    
    class_name = "Gastric cancer" if class_index == 0 else "Non-gastric cancer"
    print(f"Explaining prediction for: {class_name}")
    
    # Create Explanation object for waterfall plot
    explanation_obj = shap.Explanation(
        values=waterfall_values,
        base_values=expected_value,
        data=X_test_num.iloc[sample_idx].values,
        feature_names=X_test_num.columns.tolist()
    )
    
    # Generate waterfall plot
    shap.waterfall_plot(explanation_obj, show=False)
    
    # Save waterfall plot
    waterfall_path = os.path.join(save_dir, "figures", f"shap_waterfall_sample_{sample_idx}.png")
    plt.savefig(waterfall_path, dpi=1200, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"SHAP waterfall plot saved to: {waterfall_path}")
