#!/usr/bin/env python3
import pandas as pd

class Pipeline:
    def __init__(self, model, preprocessor, feature_cols, model_name):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_cols = feature_cols
        self.model_name = model_name
    
    def predict(self, X):
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            X = X.to_frame().T
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        X = X[self.feature_cols]
        X_prep = self.preprocessor.transform(X)
        return self.model.predict(X_prep)
    
    def predict_proba(self, X):
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        X = X[self.feature_cols]
        X_prep = self.preprocessor.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_prep)
        return None
    
    def predict_single(self, data):
        pred = self.predict(data)
        return int(pred[0])
