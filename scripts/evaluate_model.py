import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score

def evaluate_and_plot(model, X_test, y_test, label_encoder, save_prefix, save_path):
    """
    Evaluate the model and plot ROC and Precision-Recall curves.
    Parameters:
    - model: Trained model to evaluate.
    - X_test: Test features.
    - y_test: True labels for the test set.
    - label_encoder: Label encoder used to encode the labels.
    - save_prefix: Prefix for saving the output files.
    - save_path: Path to save the evaluation results and figures.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    report_dict = classification_report(
        label_encoder.inverse_transform(y_test),
        label_encoder.inverse_transform(y_pred),
        output_dict=True
    )

    pd.DataFrame(report_dict).transpose().to_csv(save_path + f"{save_prefix}_classification_report.csv", index=True)

    y_proba_0 = y_proba[:, 0]
    y_proba_1 = y_proba[:, 1]

    fpr_0, tpr_0, _ = roc_curve(y_test, y_proba_0, pos_label=0)
    fpr_1, tpr_1, _ = roc_curve(y_test, y_proba_1, pos_label=1)
    precision_0, recall_0, _ = precision_recall_curve(y_test, y_proba_0, pos_label=0)
    precision_1, recall_1, _ = precision_recall_curve(y_test, y_proba_1, pos_label=1)
    pr_auc_0 = average_precision_score(y_test, y_proba_0, pos_label=0)
    pr_auc_1 = average_precision_score(y_test, y_proba_1, pos_label=1)
    auc_0 = auc(fpr_0, tpr_0)
    auc_1 = auc(fpr_1, tpr_1)

    plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.subplots_adjust(wspace=0.3)

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
    fig.savefig(f"{save_path}/figures/{save_prefix}_ROC_PR.png", dpi=1200, bbox_inches='tight')
    fig.savefig(f"{save_path}/figures/{save_prefix}_ROC_PR.pdf", bbox_inches='tight')
    # plt.show()

    # plt.show()
