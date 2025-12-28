import argparse
import pandas as pd
from src.models.trainer import load_model
from src.evaluation.metrics import calculate_metrics
from src.utils.logger import setup_logger

def evaluate_model(model_path, test_data_path):
    logger = setup_logger()
    
    # Load the trained model
    model = load_model(model_path)
    
    # Load the test data
    test_data = pd.read_csv(test_data_path)
    
    # Separate features and target
    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate evaluation metrics
    metrics = calculate_metrics(y_test, predictions)
    
    # Log the evaluation results
    logger.info("Evaluation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data CSV file.")
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.test_data_path)