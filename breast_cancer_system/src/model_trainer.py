import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score
import warnings

# Suppress warnings from optuna
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_logistic_regression(X_train, Y_train, X_val, Y_val, n_trials=25, random_state=42):
    """
    Tune Logistic Regression hyperparameters using Optuna.
    """
    def objective(trial):
        C = trial.suggest_float('C', 1e-4, 1e2, log=True)
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
        
        clf = LogisticRegression(C=C, penalty=penalty, solver=solver, random_state=random_state, max_iter=1000)
        clf.fit(X_train, Y_train)
        
        val_preds = clf.predict(X_val)
        val_probs = clf.predict_proba(X_val)[:, 1]
        
        # Optimize recall for Malignant (class 0)
        recall = recall_score(Y_val, val_preds, pos_label=0)
        roc_auc = roc_auc_score(Y_val, val_probs)
        
        return 0.5 * recall + 0.5 * roc_auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    solver = 'liblinear' if best_params['penalty'] == 'l1' else 'lbfgs'
    
    best_clf = LogisticRegression(
        C=best_params['C'],
        penalty=best_params['penalty'],
        solver=solver,
        random_state=random_state,
        max_iter=1000
    )
    best_clf.fit(X_train, Y_train)
    return best_clf, best_params

def tune_random_forest(X_train, Y_train, X_val, Y_val, n_trials=25, random_state=42):
    """
    Tune Random Forest hyperparameters using Optuna.
    """
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 250)
        max_depth = trial.suggest_int('max_depth', 3, 12)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 8)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 8)
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        clf.fit(X_train, Y_train)
        
        val_preds = clf.predict(X_val)
        val_probs = clf.predict_proba(X_val)[:, 1]
        
        # Optimize recall for Malignant (class 0)
        recall = recall_score(Y_val, val_preds, pos_label=0)
        roc_auc = roc_auc_score(Y_val, val_probs)
        
        return 0.5 * recall + 0.5 * roc_auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    best_clf = RandomForestClassifier(**study.best_params, random_state=random_state, n_jobs=-1)
    best_clf.fit(X_train, Y_train)
    return best_clf, study.best_params

def tune_xgboost(X_train, Y_train, X_val, Y_val, n_trials=25, random_state=42):
    """
    Tune XGBoost hyperparameters using Optuna.
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 250),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
            'random_state': random_state,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, Y_train)
        
        val_preds = clf.predict(X_val)
        val_probs = clf.predict_proba(X_val)[:, 1]
        
        # Optimize recall for Malignant (class 0)
        recall = recall_score(Y_val, val_preds, pos_label=0)
        roc_auc = roc_auc_score(Y_val, val_probs)
        
        return 0.5 * recall + 0.5 * roc_auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    best_clf = xgb.XGBClassifier(**study.best_params, random_state=random_state, n_jobs=-1)
    best_clf.fit(X_train, Y_train)
    return best_clf, study.best_params

def tune_keras_mlp(X_train, Y_train, X_val, Y_val, n_trials=20, epochs=None, random_state=42):
    """
    Tune Scikit-Learn MLPClassifier (Neural Network) using Optuna.
    """
    def objective(trial):
        layer1 = trial.suggest_categorical('layer1', [16, 32, 64])
        layer2 = trial.suggest_categorical('layer2', [8, 16, 32])
        hidden_layer_sizes = (layer1, layer2)
        
        alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
        
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state
        )
        clf.fit(X_train, Y_train)
        
        val_preds = clf.predict(X_val)
        val_probs = clf.predict_proba(X_val)[:, 1]
        
        # Optimize recall for Malignant (class 0)
        recall = recall_score(Y_val, val_preds, pos_label=0)
        roc_auc = roc_auc_score(Y_val, val_probs)
        
        return 0.5 * recall + 0.5 * roc_auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_hidden_sizes = (best_params['layer1'], best_params['layer2'])
    
    best_clf = MLPClassifier(
        hidden_layer_sizes=best_hidden_sizes,
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate_init'],
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state
    )
    best_clf.fit(X_train, Y_train)
    return best_clf, best_params
