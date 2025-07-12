import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score

def evaluate_model(model, X_test, y_test, le, save_path):
    """
    Evaluate the logistic regression model on the test set and save the results.
    Parameters:
    - model: trained logistic regression model.
    - X_test: DataFrame, test features.
    - y_test: Series, test labels.
    - le: LabelEncoder, fitted label encoder for the labels.
    - save_path: str, path to save the evaluation results and figures.
    Returns:
    - None, saves the classification report and figures to the specified path.
    """

    # Create directories if they don't exist
    os.makedirs(save_path + "/figures", exist_ok=True)

    y_pred = model.predict(X_test)
    report = classification_report(
        le.inverse_transform(y_test),
        le.inverse_transform(y_pred),
        output_dict=True
    )
    pd.DataFrame(report).transpose().to_csv(save_path + "/LR_classification_report.csv")

    y_proba = model.predict_proba(X_test)
    fpr_0, tpr_0, _ = roc_curve(y_test, y_proba[:, 0], pos_label=0)
    fpr_1, tpr_1, _ = roc_curve(y_test, y_proba[:, 1], pos_label=1)
    auc_0 = auc(fpr_0, tpr_0)
    auc_1 = auc(fpr_1, tpr_1)

    precision_0, recall_0, _ = precision_recall_curve(y_test, y_proba[:, 0], pos_label=0)
    precision_1, recall_1, _ = precision_recall_curve(y_test, y_proba[:, 1], pos_label=1)

    pr_auc_0 = average_precision_score(y_test, y_proba[:, 0], pos_label=0)
    pr_auc_1 = average_precision_score(y_test, y_proba[:, 1], pos_label=1)

    plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.subplots_adjust(wspace=0.2)

    # ROC
    ax1.plot(fpr_0, tpr_0, label=f"Gastric AUC = {auc_0:.2f}", color='blue')
    ax1.plot(fpr_1, tpr_1, label=f"Non-Gastric AUC = {auc_1:.2f}", color='red')
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()

    # PR
    ax2.plot(recall_0, precision_0, label=f"Gastric PR AUC = {pr_auc_0:.2f}", color='green')
    ax2.plot(recall_1, precision_1, label=f"Non-Gastric PR AUC = {pr_auc_1:.2f}", color='orange')
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()

    # Save
    fig.savefig(f"{save_path}/figures/LR_ROC_PR.png", dpi=1200, bbox_inches='tight')
    fig.savefig(f"{save_path}/figures/LR_ROC_PR.pdf", bbox_inches='tight')
    # plt.show()
