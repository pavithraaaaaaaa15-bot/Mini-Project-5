import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from src import app as appmod
from src.app import app
import joblib
import json
import pandas as pd


def setup_module():
    try:
        appmod.model = joblib.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_model_pipeline.joblib'))
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model_metadata.json')) as f:
            appmod.metadata = json.load(f)
        appmod.feature_columns = appmod.metadata.get('feature_columns', [])
    except Exception as e:
        pass


def test_root():
    client = TestClient(app)
    response = client.get('/')
    assert response.status_code == 200


def test_health():
    client = TestClient(app)
    response = client.get('/health')
    assert response.status_code == 200


def test_model_info():
    client = TestClient(app)
    response = client.get('/model-info')
    assert response.status_code == 200


def test_predict_valid():
    client = TestClient(app)
    df = pd.read_csv('data/loan_data_processed.csv')
    feature_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                    'Loan_Amount_Term', 'Credit_History', 'Dependents',
                    'Gender', 'Married', 'Education', 'Self_Employed',
                    'Property_Area', 'Total_Income']
    features = df[feature_cols].iloc[0].to_dict()
    response = client.post('/predict', json={'features': features})
    assert response.status_code == 200
    data = response.json()
    assert 'prediction' in data
    assert 'loan_status' in data
