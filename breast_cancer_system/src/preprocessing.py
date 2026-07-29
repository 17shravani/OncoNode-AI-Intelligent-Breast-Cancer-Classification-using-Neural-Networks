import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_preprocess_data(test_size=0.2, val_size=0.1, random_state=42, save_scaler_path=None):
    """
    Loads breast cancer dataset, scales features using StandardScaler,
    and splits into train, validation, and test datasets.
    """
    # Load dataset from sklearn
    cancer = sklearn.datasets.load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['label'] = cancer.target
    
    # Separate features and target
    X = df.drop(columns='label')
    Y = df['label']
    
    # Split: Train+Val (90%) and Test (10% or 20% according to args)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, stratify=Y
    )
    
    # Split Train+Val into Train and Val
    # Calculate adjusted validation size
    val_adj_size = val_size / (1 - test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=val_adj_size, random_state=random_state, stratify=Y_train_val
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)
    
    # Save the scaler if path is provided
    if save_scaler_path:
        os.makedirs(os.path.dirname(save_scaler_path), exist_ok=True)
        joblib.dump(scaler, save_scaler_path)
        print(f"Scaler saved successfully to {save_scaler_path}")
        
    return {
        'X_train': X_train_std,
        'X_val': X_val_std,
        'X_test': X_test_std,
        'Y_train': Y_train.values,
        'Y_val': Y_val.values,
        'Y_test': Y_test.values,
        'feature_names': list(cancer.feature_names),
        'target_names': list(cancer.target_names),
        'scaler': scaler
    }
