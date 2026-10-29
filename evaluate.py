#!/usr/bin/env python3
import os
import json
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
FIG_DIR = os.path.join(REPORTS_DIR, 'figures')
METRICS_DIR = os.path.join(REPORTS_DIR, 'metrics')


def main():
    print("=" * 60)
    print("LOAN APPROVAL PREDICTION - EVALUATION")
    print("=" * 60)
    
    df = pd.read_csv(os.path.join(DATA_DIR, 'loan_data_processed.csv'))
    feature_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                    'Loan_Amount_Term', 'Credit_History', 'Dependents',
                    'Gender', 'Married', 'Education', 'Self_Employed',
                    'Property_Area', 'Total_Income']
    X = df[feature_cols]
    y = df['target']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = joblib.load(os.path.join(MODELS_DIR, 'best_model_pipeline.joblib'))
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    report = classification_report(y_test, y_pred, output_dict=True)
    eval_report = {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1_score': float(f1), 'classification_report': report}
    
    with open(os.path.join(METRICS_DIR, 'evaluation_report.json'), 'w') as f:
        json.dump(eval_report, f, indent=2)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Evaluation Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'evaluation_confusion_matrix.png'))
    plt.close()
    
    print(f"\nEvaluation complete. Report saved to {METRICS_DIR}/evaluation_report.json")
    print("=" * 60)


if __name__ == '__main__':
    main()
