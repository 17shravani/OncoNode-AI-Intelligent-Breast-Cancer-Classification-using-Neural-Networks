import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
)
import os
import shap

def evaluate_model(model, X_test, Y_test, model_name="Model"):
    """
    Computes all standard classification metrics for a model.
    Metrics (Precision, Recall, F1) are calculated with pos_label=0 (Malignant).
    """
    preds = model.predict(X_test)
    
    # Handle classification labels and probabilities
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = preds
        
    accuracy = accuracy_score(Y_test, preds)
    # Medical classification standard: pos_label=0 (Malignant class)
    precision = precision_score(Y_test, preds, pos_label=0)
    recall = recall_score(Y_test, preds, pos_label=0)
    f1 = f1_score(Y_test, preds, pos_label=0)
    
    # ROC-AUC is class-symmetric but we need to compute it based on malignant probability.
    # Note: predict_proba[:, 1] yields probability of class 1 (Benign).
    # For ROC-AUC, whether we use prob(1) or prob(0), the score is identical.
    roc_auc = roc_auc_score(Y_test, probs)
    
    # Calculate Precision-Recall Area Under Curve for class 0 (Malignant)
    # The probability of class 0 is 1 - probs
    probs_malignant = 1.0 - probs
    prec_precision, prec_recall, _ = precision_recall_curve(Y_test, probs_malignant, pos_label=0)
    pr_auc = auc(prec_recall, prec_precision)
    
    metrics = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc)
    }
    
    return metrics, preds, probs

def plot_confusion_matrix(Y_test, Y_pred, save_path, model_name="Model"):
    """
    Plots confusion matrix and saves to path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(Y_test, Y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=['Malignant (0)', 'Benign (1)'],
        yticklabels=['Malignant (0)', 'Benign (1)']
    )
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curves(models_probs_dict, Y_test, save_path):
    """
    Plots ROC curves for multiple models on a single graph and saves.
    models_probs_dict format: {model_name: test_probabilities}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    # Plot chance line
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
    
    for name, probs in models_probs_dict.items():
        fpr, tpr, _ = roc_curve(Y_test, probs)
        roc_auc = roc_auc_score(Y_test, probs)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
        
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Operating Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def generate_shap_summary(model, X_train, feature_names, save_path, model_name="Model"):
    """
    Generates SHAP summary plot for model and saves.
    Only works on Tree Models or Linear Models easily.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    explainer = None
    try:
        # Determine appropriate explainer
        if 'RandomForest' in str(type(model)) or 'XGB' in str(type(model)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
            
            # Handling RF multiclass outputs shape or XGB binary outputs shape
            if isinstance(shap_values, list):
                # For RF binary classification, index 0 corresponds to Malignant class
                shap_values = shap_values[0]
            elif len(shap_values.shape) == 3: # multi-output
                shap_values = shap_values[:, :, 0]
        elif 'LogisticRegression' in str(type(model)):
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_train)
        
        if explainer is not None:
            shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
            plt.title(f'SHAP Feature Influence (Malignant Class) - {model_name}', fontsize=14, pad=15)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"SHAP summary plot successfully saved to {save_path}")
        else:
            print("Model type not supported for automatic SHAP plotting.")
    except Exception as e:
        print(f"Failed to generate SHAP summary plot: {e}")
        plt.close()
