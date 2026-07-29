import joblib
import os
import numpy as np

# Resolve path relative to project structure (navigate up to workspace root)
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(FILE_DIR, "..", "..", "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "breast_cancer_system", "models")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")

class PredictorService:
    def __init__(self):
        self.scaler = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        try:
            if os.path.exists(SCALER_PATH) and os.path.exists(MODEL_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                self.model = joblib.load(MODEL_PATH)
                print(f"Prediction model loaded successfully from {MODEL_PATH}")
            else:
                print("Warning: Model artifacts not found at default paths. Using self-healing fallback predictor.")
        except Exception as e:
            print(f"Error loading model artifacts: {e}. Utilizing fallback predictor.")
            
    def predict(self, features_dict: dict) -> dict:
        """
        Features dict should map the 30 continuous clinical features.
        """
        feature_names = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
            'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
            'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 
            'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 
            'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ]
        
        # Assemble feature array in the correct order
        try:
            features = [features_dict.get(name, 0.0) for name in feature_names]
            X = np.array(features).reshape(1, -1)
            
            if self.scaler is not None and self.model is not None:
                # Perform standard scaling and run model prediction
                X_std = self.scaler.transform(X)
                pred_class = int(self.model.predict(X_std)[0])
                probs = self.model.predict_proba(X_std)[0]
                
                return {
                    "prediction_class": pred_class, # 0 = Malignant, 1 = Benign
                    "malignancy_prob": float(probs[0]),
                    "benignity_prob": float(probs[1]),
                    "model_used": type(self.model).__name__,
                    "status": "success"
                }
        except Exception as e:
            print(f"Error during standard model inference: {e}")
            
        # Fallback heuristic prediction (Self-healing layer)
        # Using basic rules of radius & area to estimate malignancy if model fails
        mean_radius = features_dict.get('mean radius', 14.0)
        mean_concavity = features_dict.get('mean concavity', 0.08)
        
        # High radius & concavity points to malignancy (class 0)
        if mean_radius > 15.0 or mean_concavity > 0.12:
            malignancy_prob = 0.85
            pred_class = 0
        else:
            malignancy_prob = 0.15
            pred_class = 1
            
        return {
            "prediction_class": pred_class,
            "malignancy_prob": malignancy_prob,
            "benignity_prob": 1.0 - malignancy_prob,
            "model_used": "Fallback-Heuristic-Rule-Engine",
            "status": "fallback"
        }

predictor = PredictorService()
