import argparse
import pandas as pd
from src.data.load_data import load_data
from src.data.sampling import handle_class_imbalance
from src.models.trainer import train_model
from src.utils.config import Config
from src.utils.logger import setup_logger

def main(config_path):
    # Set up logging
    logger = setup_logger()

    # Load configuration
    config = Config(config_path)

    # Load data
    logger.info("Loading data...")
    data = load_data(config.data_path)

    # Handle class imbalance
    logger.info("Handling class imbalance...")
    data_balanced = handle_class_imbalance(data, config.imbalance_strategy)

    # Train model
    logger.info("Training model...")
    model = train_model(data_balanced, config.model_params)

    logger.info("Model training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a fraud detection model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)