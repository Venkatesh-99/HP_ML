import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve, brier_score_loss, 
    precision_score, recall_score, f1_score, accuracy_score, 
    average_precision_score, classification_report
)

from sklearn.calibration import calibration_curve

plt.rcParams.update({           
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'figure.dpi': 1000, 'savefig.dpi': 1000
})

def get_class_info(le, pos_class="Gastric cancer"):
    """Figure out which label is which class"""
    mapping = {name: i for i, name in enumerate(le.classes_)}
    neg_class = [k for k in mapping.keys() if k != pos_class][0]
    
    return {
        'pos_name': pos_class, 'pos_label': mapping[pos_class],
        'neg_name': neg_class, 'neg_label': mapping[neg_class],
        'mapping': mapping
    }

def bootstrap_ci(y_true, y_pred, y_prob, metric_func, n=1000):
    """Get confidence intervals via bootstrap"""
    np.random.seed(26)  # reproducibility
    scores = []
    
    for _ in range(n):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        
        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true[idx])) < 2:
            continue
            
        try:
            if y_prob is not None:
                score = metric_func(y_true[idx], y_prob[idx])
            else:
                score = metric_func(y_true[idx], y_pred[idx])
            scores.append(score)
        except:
            continue  # skip problematic samples
    
    if not scores:
        return {'mean': np.nan, 'ci_low': np.nan, 'ci_high': np.nan}
    
    scores = np.array(scores)
    return {
        'mean': np.mean(scores),
        'ci_low': np.percentile(scores, 2.5),
        'ci_high': np.percentile(scores, 97.5)
    }

def plot_confusion_matrix(y_test, y_pred, class_info, title="Confusion Matrix"):
    """Basic confusion matrix plot"""
    labels = [class_info['neg_label'], class_info['pos_label']]  
    names = [class_info['neg_name'], class_info['pos_name']]
    
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_pct = cm.astype('float') / cm.sum() * 100
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=False, cmap='Blues', 
                xticklabels=names, yticklabels=names, ax=ax)
    
    # Add count + percentage annotations
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j + 0.5, i + 0.5, f'{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)', 
                   ha='center', va='center', color=color, fontweight='bold')
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Actual Labels')
    
    # Quick metrics for subtitle
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / cm.sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    subtitle = f'Accuracy: {acc:.3f} | Sensitivity: {sens:.3f} | Specificity: {spec:.3f}'
    ax.text(0.5, -0.12, subtitle, transform=ax.transAxes, ha='center', style='italic')
    
    return fig

def plot_roc_pr(y_test, y_prob, class_info, auc_ci, ap_ci, title="ROC & PR Curves"):
    """ROC and PR curves side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    pos_label = class_info['pos_label']
    pos_name = class_info['pos_name']
    
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, pos_label], pos_label=pos_label)
    ax1.plot(fpr, tpr, 'r-', linewidth=2,
             label=f"{pos_name} - AUC = {auc_ci['mean']:.3f} (95% CI: {auc_ci['ci_low']:.3f}–{auc_ci['ci_high']:.3f})")
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate') 
    ax1.set_title('ROC Curve')
    ax1.legend()
    
    # PR
    prec, rec, _ = precision_recall_curve(y_test, y_prob[:, pos_label], pos_label=pos_label)
    ax2.plot(rec, prec, 'b-', linewidth=2,
             label=f"{pos_name} - AP = {ap_ci['mean']:.3f} (95% CI: {ap_ci['ci_low']:.3f}–{ap_ci['ci_high']:.3f})")
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_calibration(y_test, y_prob_cal, y_prob_raw, class_info, title="Calibration"):
    """Before/after calibration plots"""
    pos_label = class_info['pos_label']
    y_true_binary = (y_test == pos_label).astype(int)
    
    # FIXED: Handle axes creation properly
    if y_prob_raw is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        axes = [ax]
    
    plot_idx = 0
    
    # Before calibration
    if y_prob_raw is not None:
        frac, pred = calibration_curve(y_true_binary, y_prob_raw[:, pos_label], n_bins=10)
        brier = brier_score_loss(y_true_binary, y_prob_raw[:, pos_label])
        
        axes[plot_idx].plot(pred, frac, 's-', label=class_info['pos_name'])
        axes[plot_idx].plot([0, 1], [0, 1], 'k:', label="Perfect")
        axes[plot_idx].set_title("Before Calibration")
        axes[plot_idx].text(0.02, 0.98, f'Brier Score: {brier:.3f}', 
                           transform=axes[plot_idx].transAxes, va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[plot_idx].legend()
        plot_idx += 1
    
    # After calibration
    frac, pred = calibration_curve(y_true_binary, y_prob_cal[:, pos_label], n_bins=10)
    brier = brier_score_loss(y_true_binary, y_prob_cal[:, pos_label])
    
    axes[plot_idx].plot(pred, frac, 's-', label=class_info['pos_name'])
    axes[plot_idx].plot([0, 1], [0, 1], 'k:', label="Perfect")
    axes[plot_idx].set_title("After Calibration" if y_prob_raw is not None else "Calibration")
    axes[plot_idx].text(0.02, 0.98, f'Brier Score: {brier:.3f}', 
                       transform=axes[plot_idx].transAxes, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[plot_idx].legend()
    
    # FIXED: Now axes is always a list, so this works correctly
    for ax in axes:
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_metrics_bar(y_test, y_pred, class_info, title="Classification Metrics"):
    """Bar plot of precision, recall, F1 with confidence intervals"""
    pos_label = class_info['pos_label']
    neg_label = class_info['neg_label']
    
    # Calculate metrics with CIs for both classes
    metrics = {}
    for label, name in [(pos_label, class_info['pos_name']), (neg_label, class_info['neg_name'])]:
        metrics[name] = {
            'Precision': bootstrap_ci(y_test, y_pred, None, lambda yt, yp: precision_score(yt, yp, pos_label=label, zero_division=0)),
            'Recall': bootstrap_ci(y_test, y_pred, None, lambda yt, yp: recall_score(yt, yp, pos_label=label, zero_division=0)),
            'F1-Score': bootstrap_ci(y_test, y_pred, None, lambda yt, yp: f1_score(yt, yp, pos_label=label, zero_division=0))
        }
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metric_names = ['Precision', 'Recall', 'F1-Score']
    class_names = list(metrics.keys())
    
    # Get data
    pos_means = [metrics[class_names[0]][m]['mean'] for m in metric_names]
    neg_means = [metrics[class_names[1]][m]['mean'] for m in metric_names]
    
    pos_errs = [(metrics[class_names[0]][m]['ci_high'] - metrics[class_names[0]][m]['ci_low'])/2 for m in metric_names]
    neg_errs = [(metrics[class_names[1]][m]['ci_high'] - metrics[class_names[1]][m]['ci_low'])/2 for m in metric_names]
    
    # Bar positions
    y_pos = np.arange(len(metric_names))
    width = 0.35
    
    # Create bars
    bars1 = ax.barh(y_pos - width/2, pos_means, width, xerr=pos_errs, 
                    label=class_names[0], color='#E31A1C', alpha=0.8, capsize=3, hatch='///')
    bars2 = ax.barh(y_pos + width/2, neg_means, width, xerr=neg_errs, 
                    label=class_names[1], color='#1F78B4', alpha=0.8, capsize=3, hatch='...')
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax.text(bar1.get_width() + pos_errs[i] + 0.02, bar1.get_y() + bar1.get_height()/2,
                f'{pos_means[i]:.3f}', ha='left', va='center', fontsize=8)
        ax.text(bar2.get_width() + neg_errs[i] + 0.02, bar2.get_y() + bar2.get_height()/2,
                f'{neg_means[i]:.3f}', ha='left', va='center', fontsize=8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_names)
    ax.set_xlabel('Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 1.15)
    
    plt.tight_layout()
    return fig


def create_metrics_table(y_test, y_pred, y_prob, class_info):
    """Create comprehensive metrics table with confidence intervals"""
    pos_label = class_info['pos_label']
    neg_label = class_info['neg_label']
    pos_name = class_info['pos_name']
    neg_name = class_info['neg_name']
    
    # Calculate metrics with CIs for both classes
    metrics_data = []
    
    # Per-class metrics
    for label, class_name in [(pos_label, pos_name), (neg_label, neg_name)]:
        # Precision
        prec_ci = bootstrap_ci(y_test, y_pred, None, 
                              lambda yt, yp: precision_score(yt, yp, pos_label=label, zero_division=0))
        # Recall (Sensitivity for pos class, Specificity calculation for neg class)
        rec_ci = bootstrap_ci(y_test, y_pred, None, 
                             lambda yt, yp: recall_score(yt, yp, pos_label=label, zero_division=0))
        # F1-Score
        f1_ci = bootstrap_ci(y_test, y_pred, None, 
                            lambda yt, yp: f1_score(yt, yp, pos_label=label, zero_division=0))
        
        metrics_data.extend([
            {
                'Class': class_name,
                'Metric': 'Precision',
                'Value': prec_ci['mean'],
                'CI_Lower': prec_ci['ci_low'],
                'CI_Upper': prec_ci['ci_high'],
                'Final': f"{prec_ci['mean']:.3f} ({prec_ci['ci_low']:.3f}–{prec_ci['ci_high']:.3f})"
            },
            {
                'Class': class_name,
                'Metric': 'Recall' if label == pos_label else 'Specificity',
                'Value': rec_ci['mean'],
                'CI_Lower': rec_ci['ci_low'],
                'CI_Upper': rec_ci['ci_high'],
                'Final': f"{rec_ci['mean']:.3f} ({rec_ci['ci_low']:.3f}–{rec_ci['ci_high']:.3f})"
            },
            {
                'Class': class_name,
                'Metric': 'F1-Score',
                'Value': f1_ci['mean'],
                'CI_Lower': f1_ci['ci_low'],
                'CI_Upper': f1_ci['ci_high'],
                'Final': f"{f1_ci['mean']:.3f} ({f1_ci['ci_low']:.3f}–{f1_ci['ci_high']:.3f})"
            }
        ])
    
    # Overall metrics
    # Accuracy
    acc_ci = bootstrap_ci(y_test, y_pred, None, lambda yt, yp: accuracy_score(yt, yp))
    
    # AUC
    def auc_func(y_true, y_prob):
        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=pos_label)
        return auc(fpr, tpr)
    auc_ci = bootstrap_ci(y_test, None, y_prob[:, pos_label], auc_func)
    
    # Average Precision
    ap_ci = bootstrap_ci(y_test, None, y_prob[:, pos_label], 
                        lambda yt, yp: average_precision_score(yt, yp, pos_label=pos_label))
    
    # Add overall metrics
    overall_metrics = [
        {
            'Class': 'Overall',
            'Metric': 'Accuracy',
            'Value': acc_ci['mean'],
            'CI_Lower': acc_ci['ci_low'],
            'CI_Upper': acc_ci['ci_high'],
            'Final': f"{acc_ci['mean']:.3f} ({acc_ci['ci_low']:.3f}–{acc_ci['ci_high']:.3f})"
        },
        {
            'Class': 'Overall',
            'Metric': 'AUC-ROC',
            'Value': auc_ci['mean'],
            'CI_Lower': auc_ci['ci_low'],
            'CI_Upper': auc_ci['ci_high'],
            'Final': f"{auc_ci['mean']:.3f} ({auc_ci['ci_low']:.3f}–{auc_ci['ci_high']:.3f})"
        },
        {
            'Class': 'Overall',
            'Metric': 'Average Precision',
            'Value': ap_ci['mean'],
            'CI_Lower': ap_ci['ci_low'],
            'CI_Upper': ap_ci['ci_high'],
            'Final': f"{ap_ci['mean']:.3f} ({ap_ci['ci_low']:.3f}–{ap_ci['ci_high']:.3f})"
        }
    ]
    
    metrics_data.extend(overall_metrics)
    
    # Create DataFrame
    df = pd.DataFrame(metrics_data)
    
    return df


def evaluate_and_plot(model, X_test, y_test, le, save_dir, prefix, 
               model_name="Model", raw_model=None):
    """Main evaluation function
    Parameters:
    - model: trained model (e.g., RandomForest, LogisticRegression, XGBoost).
    - X_test: DataFrame, test features.
    - y_test: Series, test labels.
    - le: LabelEncoder, fitted label encoder for the labels.
    - save_dir: str, directory to save results and figures.
    - prefix: str, prefix for saved files.
    - model_name: str, name of the model for display purposes.
    - raw_model: optional, raw model for calibration plots if applicable.
    
    Returns:
    - Dictionary with evaluation metrics and class info.
    """
    
    os.makedirs(f"{save_dir}/figures", exist_ok=True)
    
    class_info = get_class_info(le)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    y_prob_raw = raw_model.predict_proba(X_test) if raw_model else None
    
    # Classification report
    y_test_orig = le.inverse_transform(y_test)
    y_pred_orig = le.inverse_transform(y_pred)
    report = classification_report(y_test_orig, y_pred_orig, output_dict=True)
    pd.DataFrame(report).T.to_csv(f"{save_dir}/{prefix}_report.csv")
    
    # Calculate metrics with CIs
    pos_label = class_info['pos_label']
    
    def auc_func(y_true, y_prob):
        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=pos_label)
        return auc(fpr, tpr)
    
    def ap_func(y_true, y_prob):
        return average_precision_score(y_true, y_prob, pos_label=pos_label)
    
    auc_ci = bootstrap_ci(y_test, None, y_prob[:, pos_label], auc_func)
    ap_ci = bootstrap_ci(y_test, None, y_prob[:, pos_label], ap_func)

    metrics_table = create_metrics_table(y_test, y_pred, y_prob, class_info)
    
    # Save the table
    metrics_table.to_csv(f"{save_dir}/{prefix}_detailed_metrics.csv", index=False)
    
    # Create plots
    fig1 = plot_confusion_matrix(y_test, y_pred, class_info, f"{model_name} Confusion Matrix")
    fig2 = plot_roc_pr(y_test, y_prob, class_info, auc_ci, ap_ci, f"{model_name} Curves")
    fig3 = plot_metrics_bar(y_test, y_pred, class_info, f"{model_name} Classification Metrics")
    fig4 = plot_calibration(y_test, y_prob, y_prob_raw, class_info, f"{model_name} Calibration")
    
    # Save plots
    plots = [(fig1, f"{prefix}_confusion"), (fig2, f"{prefix}_curves"), 
             (fig3, f"{prefix}_metrics"), (fig4, f"{prefix}_calibration")]
    
    for fig, name in plots:
        fig.savefig(f"{save_dir}/figures/{name}.png", dpi=1200, bbox_inches='tight')
        fig.savefig(f"{save_dir}/figures/{name}.pdf", bbox_inches='tight')
        # plt.show()
    
    print(f"Saved plots to {save_dir}/figures/")
    
    return {
        'auc': auc_ci, 'ap': ap_ci, 'class_info': class_info,
        'report': report
    }

