import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd


def test_data_files_exist():
    assert os.path.exists('data/loan_data_raw.csv')
    assert os.path.exists('data/loan_data_processed.csv')


def test_data_loaded():
    df = pd.read_csv('data/loan_data_processed.csv')
    assert len(df) > 0
    assert 'target' in df.columns
    assert df.isnull().sum().sum() == 0
