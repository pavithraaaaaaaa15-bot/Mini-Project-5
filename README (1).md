# Loan Approval Prediction System

## Project Overview

End-to-end Machine Learning system for predicting loan approval status based on applicant information. Built with production-ready architecture including MLflow tracking, FastAPI REST API, Docker containerization, and automated testing.

**Problem Type:** Binary Classification  
**Target Variable:** Loan_Status (Approved/Rejected)  
**Use Case:** Automated loan screening, credit risk assessment

## Features

- **Data Processing**: Missing value handling, categorical encoding, feature scaling
- **Multiple Models**: Logistic Regression, Random Forest, Gradient Boosting
- **MLflow Tracking**: Experiment tracking and model management
- **Production API**: FastAPI with validation and probability outputs
- **Docker Ready**: Containerized deployment
- **Comprehensive Testing**: Pytest test suite

## Tech Stack

- **Data:** pandas, numpy
- **ML:** scikit-learn, xgboost, lightgbm, imbalanced-learn
- **Visualization:** matplotlib, seaborn
- **MLOps:** MLflow
- **API:** FastAPI, uvicorn, pydantic
- **Testing:** pytest

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **RandomForest (Best)** | 0.9024 | 0.9042 | 0.9024 | 0.9024 |
| GradientBoosting | 0.8780 | 0.8792 | 0.8780 | 0.8780 |
| LogisticRegression | 0.7724 | 0.7727 | 0.7724 | 0.7723 |

## Quick Start

```bash
# Install dependencies
py -m pip install -r requirements.txt

# Data preparation
py src/data_preparation.py

# Train
set PYTHONPATH=. && set MLFLOW_TRACKING_URI=file:./mlruns && py src/train2.py

# Evaluate
set PYTHONPATH=. && py src/evaluate.py

# Test
set PYTHONPATH=. && py -m pytest tests/ -v

# Serve
set PYTHONPATH=. && py -m uvicorn src.app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- GET `/` - API info
- GET `/health` - Health check
- GET `/model-info` - Model metadata
- POST `/predict` - Predict loan approval

## Docker

```bash
docker build -t loan-approval-prediction:latest .
docker run -d -p 8000:8000 loan-approval-prediction:latest
```

## Known Limitations

- Synthetic dataset used (clearly identified)
- For educational purposes
- Model predictions not a substitute for actual lending decisions

## License

Educational/Capstone project.
