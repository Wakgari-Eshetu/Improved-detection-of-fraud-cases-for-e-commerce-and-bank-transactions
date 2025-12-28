import pandas as pd
from src.data.load_data import load_raw_data
from src.preprocessing.clean import clean_data
from src.preprocessing.impute import impute_missing_values
from src.preprocessing.transform import transform_features
from src.data.sampling import handle_class_imbalance

def preprocess_data():
    # Load raw data
    raw_data = load_raw_data()

    # Clean the data
    cleaned_data = clean_data(raw_data)

    # Impute missing values
    imputed_data = impute_missing_values(cleaned_data)

    # Transform features (scaling, encoding, etc.)
    transformed_data = transform_features(imputed_data)

    # Handle class imbalance
    balanced_data = handle_class_imbalance(transformed_data)

    # Save the processed data
    balanced_data.to_csv('data/processed/balanced_data.csv', index=False)

if __name__ == "__main__":
    preprocess_data()