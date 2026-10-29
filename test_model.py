import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
import json


def test_model_artifacts_exist():
    assert os.path.exists('models/best_model_pipeline.joblib')
    assert os.path.exists('models/best_model.joblib')
    assert os.path.exists('models/preprocessor.joblib')
    assert os.path.exists('models/model_metadata.json')


def test_model_loads():
    model = joblib.load('models/best_model_pipeline.joblib')
    assert model is not None


def test_metadata_valid():
    with open('models/model_metadata.json') as f:
        meta = json.load(f)
    assert 'best_model_name' in meta
    assert 'metrics' in meta
    assert 'accuracy' in meta['metrics']
