#!/usr/bin/env python3
import os
import joblib
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

class LoanPredictor:
    def __init__(self):
        self.model = joblib.load(os.path.join(MODELS_DIR, 'best_model_pipeline.joblib'))
        with open(os.path.join(MODELS_DIR, 'model_metadata.json')) as f:
            self.metadata = json.load(f)
    
    def predict(self, features):
        pred = self.model.predict(features)
        proba = self.model.predict_proba(features)
        return {'prediction': int(pred[0]), 'probabilities': proba[0].tolist() if proba is not None else None}

if __name__ == '__main__':
    predictor = LoanPredictor()
    print("Model loaded successfully")
    print(predictor.metadata)
