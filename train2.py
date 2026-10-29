#!/usr/bin/env python3
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from src.pipeline import Pipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MLRUNS_DIR = os.path.join(PROJECT_ROOT, 'mlruns')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
FIG_DIR = os.path.join(REPORTS_DIR, 'figures')
METRICS_DIR = os.path.join(REPORTS_DIR, 'metrics')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
mlflow.set_tracking_uri('file:' + os.path.abspath(MLRUNS_DIR))
mlflow.set_experiment('loan_approval_prediction')


def main():
    print("=" * 60)
    print("LOAN APPROVAL PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    df = pd.read_csv(os.path.join(DATA_DIR, 'loan_data_processed.csv'))
    
    feature_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                    'Loan_Amount_Term', 'Credit_History', 'Dependents',
                    'Gender', 'Married', 'Education', 'Self_Employed',
                    'Property_Area', 'Total_Income']
    
    X = df[feature_cols]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', 'passthrough', categorical_cols)
    ])
    
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param('model_name', name)
            model.fit(X_train_prep, y_train)
            y_pred = model.predict(X_test_prep)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            mlflow.log_metric('accuracy', acc)
            mlflow.log_metric('f1_score', f1)
            print(f"\n{name}: Acc={acc:.4f}, F1={f1:.4f}")
            results.append({'model': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'model_obj': model, 'y_pred': y_pred})
    
    best = max(results, key=lambda x: x['f1_score'])
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best['model']} (F1: {best['f1_score']:.4f})")
    print(f"{'='*60}")
    
    pipeline = Pipeline(best['model_obj'], preprocessor, feature_cols, best['model'])
    joblib.dump(pipeline, os.path.join(MODELS_DIR, 'best_model_pipeline.joblib'))
    joblib.dump(best['model_obj'], os.path.join(MODELS_DIR, 'best_model.joblib'))
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    
    metadata = {
        'best_model_name': best['model'],
        'feature_columns': feature_cols,
        'metrics': {'accuracy': float(best['accuracy']), 'precision': float(best['precision']), 'recall': float(best['recall']), 'f1_score': float(best['f1_score'])},
        'timestamp': datetime.now().isoformat(),
        'model_type': 'classification'
    }
    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    with open(os.path.join(METRICS_DIR, 'model_comparison.json'), 'w') as f:
        json.dump([{'model': r['model'], 'accuracy': float(r['accuracy']), 'f1_score': float(r['f1_score'])} for r in results], f, indent=2)
    
    cm = confusion_matrix(y_test, best['y_pred'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best["model"]}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'confusion_matrix.png'))
    plt.close()
    
    print(f"\nModel saved to {MODELS_DIR}/best_model_pipeline.joblib")
    print("=" * 60)


if __name__ == '__main__':
    main()
