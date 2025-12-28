# scripts/train.py

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.trainer import train_and_evaluate
from src.utils.config import Config
from src.utils.logger import setup_logger
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_model(input_dim):
    """Builds a simple Keras model for binary classification."""
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main(config_path):
    # Set up logging
    logger = setup_logger()

    # Load configuration
    config = Config(config_path)

    # -------------------------------
    # Load ready-for-training data
    # -------------------------------
    logger.info("Loading ready-for-training data...")
    df = pd.read_csv(config.data_path)  # config.data_path should point to Fraud_Data_ready.csv

    target_col = config.target_col
    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(f"Data shape: {X.shape}, Target shape: {y.shape}")

    # -------------------------------
    # Optionally handle class imbalance again if needed
    # (usually already done in build_features)
    # -------------------------------
    if getattr(config, "rebalance", False):
        logger.info("Applying SMOTE for class imbalance...")
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        logger.info(f"Shape after SMOTE: {X.shape}")

    # -------------------------------
    # Train model
    # -------------------------------
    logger.info("Training model...")
    model = train_model(X, y, config.model_params)

    logger.info("Model training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a fraud detection model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)
