# Fraud Detection Project

This project aims to improve fraud detection in e-commerce and bank transactions using machine learning techniques. The focus is on handling class imbalance and providing explainability for the models used.

## Project Structure

- **notebooks/**: Contains Jupyter notebooks for exploratory data analysis, data preprocessing, feature engineering, modeling, and explainability.
  - `00-eda-fraud-data.ipynb`: Exploratory Data Analysis (EDA) on the fraud dataset.
  - `01-data-preprocessing.ipynb`: Data cleaning and preprocessing tasks.
  - `02-feature-engineering.ipynb`: Feature engineering techniques and handling class imbalance.
  - `03-modeling.ipynb`: Building and training machine learning models.
  - `04-explainability.ipynb`: Interpreting model predictions using SHAP and LIME.

- **src/**: Source code for data handling, preprocessing, feature engineering, model training, evaluation, and explainability.
  - **data/**: Functions for loading and managing datasets.
  - **preprocessing/**: Functions for cleaning, imputing, and transforming data.
  - **features/**: Functions for building and scaling features.
  - **models/**: Functions for training models and defining architectures.
  - **evaluation/**: Functions for calculating metrics and cross-validation.
  - **explainability/**: Functions for generating SHAP and LIME explanations.
  - **utils/**: Utility functions for configuration and logging.

- **experiments/**: Configuration and logs for different experiment runs.
  - `config.yaml`: Hyperparameter settings for experiments.
  - `runs/`: Directory for storing results and logs.

- **data/**: Directory structure for raw, interim, and processed datasets.
  - **raw/**: Original datasets.
  - **interim/**: Intermediate datasets.
  - **processed/**: Fully processed datasets ready for modeling.

- **scripts/**: Scripts for running the preprocessing, training, evaluation, and explanation pipelines.
  - `preprocess.py`: Runs the preprocessing pipeline.
  - `train.py`: Trains machine learning models.
  - `evaluate.py`: Evaluates trained models.
  - `explain.py`: Generates explanations for model predictions.

- **tests/**: Unit tests for data and model functions.
  - `test_data.py`: Tests for data loading and preprocessing.
  - `test_models.py`: Tests for model training and evaluation.

- **docs/**: Documentation for the project.
  - `MODEL_CARD.md`: Summary of model performance and intended use.
  - `EXPERIMENTS.md`: Details of experiments conducted.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd fraud-detection-project
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up the environment:
   ```
   conda env create -f environment.yml
   conda activate <environment-name>
   ```

## Usage Guidelines

- Use the Jupyter notebooks for interactive analysis and development.
- Run the scripts in the `scripts/` directory for batch processing tasks.
- Refer to the documentation in the `docs/` directory for detailed information on model performance and experiments.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.