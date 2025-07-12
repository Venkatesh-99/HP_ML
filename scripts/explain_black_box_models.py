import os
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def plot_shap_summary(pipeline, X, save_path, classifier_step_name='classifier', class_index=None):
    """
    Generate and save a SHAP summary plot for a tree-based model inside a pipeline.

    Parameters:
    - pipeline: sklearn Pipeline, must contain a tree-based model with SHAP support.
    - X: DataFrame, the feature data to explain.
    - save_path: str, path to save the SHAP summary plot (e.g., 'figures/shap_summary.png').
    - classifier_step_name: str, name of the classifier step in the pipeline (default: 'classifier').
    - class_index: int or None. If the model is multiclass or binary with SHAP output as a list,
                   this specifies which class's SHAP values to plot. If None, the whole array is used.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = pipeline.estimator.named_steps[classifier_step_name]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Special case for RandomForestClassifier with 3D SHAP values (e.g., [n_samples, n_features, n_classes])
    if isinstance(model, RandomForestClassifier) and hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        if class_index is None:
            raise ValueError(f"`class_index` must be specified for RandomForestClassifier with 3D SHAP output.")
        values_to_plot = shap_values[:, :, class_index]
        print(f"Plotting SHAP values for RandomForestClassifier class index {class_index} (3D SHAP values).")

    # Generic multi-class case: shap_values is a list of arrays [class_0, class_1, ...]
    elif isinstance(shap_values, list):
        n_classes = len(shap_values)
        if class_index is None:
            raise ValueError(f"`class_index` must be specified for models with multiple classes (found {n_classes}).")
        if not (0 <= class_index < n_classes):
            raise ValueError(f"Invalid class_index {class_index}; must be between 0 and {n_classes - 1}.")
        values_to_plot = shap_values[class_index]
        print(f"Plotting SHAP values for class index {class_index}.")

    # Binary or single-output models
    else:
        values_to_plot = shap_values
        if class_index is not None:
            print("Warning: class_index ignored for single-output SHAP values.")

    # Plot and save
    shap.summary_plot(values_to_plot, X, plot_size=(20, 10), plot_type="violin", show=False)
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.close()

    print(f"SHAP summary plot saved to {save_path}")
