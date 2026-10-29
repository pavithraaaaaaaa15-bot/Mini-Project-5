#!/usr/bin/env python3
import os
import json
from datetime import datetime
from typing import List
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

app = FastAPI(
    title='Loan Approval Prediction System',
    description='AI-powered loan approval prediction using machine learning',
    version='1.0.0'
)

model = None
metadata = {}
feature_columns = []


class PredictionRequest(BaseModel):
    features: dict = Field(..., description='Feature dictionary')


class PredictionResponse(BaseModel):
    prediction: int
    loan_status: str
    probabilities: List[float] = None
    confidence: float = None
    model_name: str
    timestamp: str


@app.on_event('startup')
async def load_model():
    global model, metadata, feature_columns
    try:
        model = joblib.load(os.path.join(MODELS_DIR, 'best_model_pipeline.joblib'))
        with open(os.path.join(MODELS_DIR, 'model_metadata.json')) as f:
            metadata = json.load(f)
        feature_columns = metadata.get('feature_columns', [])
        print(f"Model loaded: {metadata.get('best_model_name')}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None


@app.get('/')
async def root():
    return {
        'message': 'Loan Approval Prediction System API',
        'version': '1.0.0',
        'description': 'Predict loan approval status',
        'endpoints': ['/health', '/model-info', '/predict'],
        'docs': '/docs'
    }


@app.get('/health')
async def health():
    return {
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    }


@app.get('/model-info')
async def model_info():
    return {
        'model_name': metadata.get('best_model_name'),
        'model_type': metadata.get('model_type'),
        'feature_columns': feature_columns,
        'metrics': metadata.get('metrics'),
        'timestamp': metadata.get('timestamp')
    }


@app.post('/predict', response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    try:
        data = request.features
        pred = model.predict(data)
        prediction = int(pred[0])
        proba = model.predict_proba(data)
        probabilities = proba[0].tolist() if proba is not None else None
        confidence = float(max(probabilities)) if probabilities else None
        loan_status = "Approved" if prediction == 1 else "Rejected"
        return PredictionResponse(
            prediction=prediction,
            loan_status=loan_status,
            probabilities=probabilities,
            confidence=confidence,
            model_name=metadata.get('best_model_name', 'unknown'),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
