import os
import json
import joblib
import pandas as pd
import numpy as np
from src.preprocessing import load_and_preprocess_data
from src.model_trainer import (
    tune_logistic_regression, tune_random_forest, 
    tune_xgboost, tune_keras_mlp
)
from src.evaluator import (
    evaluate_model, plot_confusion_matrix, 
    plot_roc_curves, generate_shap_summary
)

def run_train_pipeline():
    print("==================================================")
    print("Starting Breast Cancer Classification ML Pipeline")
    print("==================================================")
    
    # 1. Load and Preprocess Data
    data_dir = "breast_cancer_system/models"
    plots_dir = "breast_cancer_system/plots"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    scaler_path = os.path.join(data_dir, "scaler.joblib")
    data = load_and_preprocess_data(test_size=0.2, val_size=0.1, random_state=42, save_scaler_path=scaler_path)
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    Y_train = data['Y_train']
    Y_val = data['Y_val']
    Y_test = data['Y_test']
    feature_names = data['feature_names']
    
    print(f"Data successfully split and standardized:")
    print(f"  Training set size  : {X_train.shape[0]}")
    print(f"  Validation set size: {X_val.shape[0]}")
    print(f"  Test set size      : {X_test.shape[0]}")
    
    # 2. Hyperparameter Tuning using Optuna
    print("\nTuning models (Optuna)...")
    
    print("  Tuning Logistic Regression...")
    lr_model, lr_params = tune_logistic_regression(X_train, Y_train, X_val, Y_val, n_trials=25)
    print(f"  Best LogReg params: {lr_params}")
    
    print("  Tuning Random Forest...")
    rf_model, rf_params = tune_random_forest(X_train, Y_train, X_val, Y_val, n_trials=25)
    print(f"  Best Random Forest params: {rf_params}")
    
    print("  Tuning XGBoost...")
    xgb_model, xgb_params = tune_xgboost(X_train, Y_train, X_val, Y_val, n_trials=25)
    print(f"  Best XGBoost params: {xgb_params}")
    
    print("  Tuning Neural Network (MLP)...")
    mlp_model, mlp_params = tune_keras_mlp(X_train, Y_train, X_val, Y_val, n_trials=20)
    print(f"  Best Neural Network (MLP) params: {mlp_params}")
    
    models = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "Neural Network (MLP)": mlp_model
    }
    
    # 3. Evaluation and Leaderboard Creation
    print("\nEvaluating models on Test Set...")
    leaderboard = []
    models_probs = {}
    
    for name, model in models.items():
        metrics, preds, probs = evaluate_model(model, X_test, Y_test, model_name=name)
        leaderboard.append(metrics)
        models_probs[name] = probs
        
        # Save individual confusion matrices
        cm_path = os.path.join(plots_dir, f"confusion_matrix_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")
        plot_confusion_matrix(Y_test, preds, cm_path, model_name=name)
        
    leaderboard_df = pd.DataFrame(leaderboard).sort_values(by='recall', ascending=False)
    print("\n--- MODEL LEADERBOARD (Sorted by Recall) ---")
    print(leaderboard_df.to_string(index=False))
    
    # Save metrics JSON
    metrics_path = os.path.join(data_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(leaderboard, f, indent=4)
    print(f"\nLeaderboard metrics saved to {metrics_path}")
    
    # 4. Generate Combined ROC Curve
    roc_path = os.path.join(plots_dir, "roc_curve_comparison.png")
    plot_roc_curves(models_probs, Y_test, roc_path)
    print(f"ROC curve comparison plot saved to {roc_path}")
    
    # 5. Generate SHAP Summary plots for top tree/linear models
    print("\nGenerating SHAP explanations...")
    shap_path_xgb = os.path.join(plots_dir, "shap_summary_xgboost.png")
    generate_shap_summary(xgb_model, X_train, feature_names, shap_path_xgb, model_name="XGBoost")
    
    shap_path_rf = os.path.join(plots_dir, "shap_summary_random_forest.png")
    generate_shap_summary(rf_model, X_train, feature_names, shap_path_rf, model_name="Random Forest")
    
    # 6. Select and Save the Best Model
    # We prioritize Recall (critical in medical diagnostics) first, then ROC-AUC
    best_row = leaderboard_df.iloc[0]
    best_model_name = best_row['model_name']
    best_model_obj = models[best_model_name]
    
    print(f"\n==================================================")
    print(f"Selecting the Best Model: {best_model_name}")
    print(f"  Recall  : {best_row['recall']:.4f}")
    print(f"  ROC-AUC : {best_row['roc_auc']:.4f}")
    print(f"  Accuracy: {best_row['accuracy']:.4f}")
    print(f"==================================================")
    
    model_metadata = {
        "best_model_name": best_model_name,
        "metrics": best_row.to_dict(),
        "params": {
            "Logistic Regression": lr_params,
            "Random Forest": rf_params,
            "XGBoost": xgb_params,
            "Neural Network (MLP)": mlp_params
        }
    }
    
    # Save best model objects
    meta_path = os.path.join(data_dir, "model_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(model_metadata, f, indent=4)
        
    # Save standard sklearn/xgboost using joblib
    sklearn_model_path = os.path.join(data_dir, "best_model.joblib")
    joblib.dump(best_model_obj, sklearn_model_path)
    print(f"Saved best model ({best_model_name}) to {sklearn_model_path}")
        
    print("\nPipeline execution complete! Models tuned and verified.")

if __name__ == "__main__":
    run_train_pipeline()
