import pytest
from src.data.load_data import load_data
from src.preprocessing.clean import clean_data
from src.preprocessing.impute import impute_missing_values
from src.preprocessing.transform import transform_data

def test_load_data():
    data = load_data('data/raw/sample_data.csv')
    assert data is not None
    assert not data.empty

def test_clean_data():
    data = load_data('data/raw/sample_data.csv')
    cleaned_data = clean_data(data)
    assert 'duplicate_column' not in cleaned_data.columns
    assert cleaned_data.isnull().sum().sum() == 0

def test_impute_missing_values():
    data = load_data('data/raw/sample_data_with_nan.csv')
    imputed_data = impute_missing_values(data)
    assert imputed_data.isnull().sum().sum() == 0

def test_transform_data():
    data = load_data('data/raw/sample_data.csv')
    transformed_data = transform_data(data)
    assert 'scaled_feature' in transformed_data.columns
    assert transformed_data['scaled_feature'].min() >= 0
    assert transformed_data['scaled_feature'].max() <= 1