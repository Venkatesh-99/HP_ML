import shap
import matplotlib.pyplot as plt

def shap_summary_plot(model, X_test, save_path):
    X_test_numeric = X_test.copy()
    for col in X_test_numeric.select_dtypes(include='category').columns:
        X_test_numeric[col] = X_test_numeric[col].cat.codes

    explainer = shap.Explainer(model.named_steps["logisticregression"], X_test_numeric)
    shap_values = explainer.shap_values(X_test_numeric)

    shap.summary_plot(shap_values, X_test_numeric, feature_names=X_test_numeric.columns.tolist(),
                      plot_size=(20, 10), plot_type="violin", show=False)

    plt.tight_layout()
    plt.savefig(save_path + "/figures/LR_shap_summary_plot.png", dpi=1200, bbox_inches="tight")
