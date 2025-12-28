# Configuration settings for the fraud detection project

import os

class Config:
    # General settings
    PROJECT_NAME = "Fraud Detection Project"
    SEED = 42

    # Data settings
    DATA_DIR = os.path.join("data", "raw")
    INTERIM_DATA_DIR = os.path.join("data", "interim")
    PROCESSED_DATA_DIR = os.path.join("data", "processed")

    # Model settings
    MODEL_DIR = os.path.join("src", "models")
    EVALUATION_METRICS = ["accuracy", "precision", "recall", "f1-score"]

    # Experiment settings
    EXPERIMENTS_DIR = os.path.join("experiments", "runs")
    CONFIG_FILE = os.path.join("experiments", "config.yaml")

    # Logging settings
    LOGGING_LEVEL = "INFO"
    LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"