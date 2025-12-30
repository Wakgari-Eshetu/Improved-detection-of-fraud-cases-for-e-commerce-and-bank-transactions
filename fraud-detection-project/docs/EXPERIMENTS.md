# Experiments Conducted for Fraud Detection Project

This document outlines the various experiments conducted during the fraud detection project, focusing on improving the detection of fraudulent transactions in e-commerce and banking. The experiments are categorized based on the tasks performed, methodologies used, and results obtained.

## 1. Data Exploration and Preprocessing

### Experiment 1: Exploratory Data Analysis (EDA)
- **Objective**: Understand the distribution of features and identify potential anomalies.
- **Methods**: Univariate and bivariate analyses, visualizations (histograms, box plots, correlation matrices).
- **Results**: Identified key features correlated with fraud, such as transaction amount and user behavior patterns.

### Experiment 2: Data Cleaning
- **Objective**: Prepare the dataset for modeling by addressing missing values and duplicates.
- **Methods**: Removal of duplicates, imputation of missing values using mean/mode, and correction of data types.
- **Results**: Cleaned dataset with improved quality for subsequent analysis.

## 2. Feature Engineering

### Experiment 3: Feature Creation
- **Objective**: Enhance model performance by creating new features.
- **Methods**: Time-based features (e.g., transaction time), transaction frequency metrics, and user behavior features.
- **Results**: Increased feature set led to improved model accuracy.

### Experiment 4: Handling Class Imbalance
- **Objective**: Address the class imbalance in the dataset.
- **Methods**: Implemented SMOTE (Synthetic Minority Over-sampling Technique) and undersampling techniques.
- **Results**: Balanced dataset improved model training and evaluation metrics.

## 3. Model Building

### Experiment 5: Baseline Model
- **Objective**: Establish a baseline for model performance.
- **Methods**: Trained a logistic regression model on the cleaned dataset.
- **Results**: Achieved baseline accuracy of X% with a recall of Y%.

### Experiment 6: Ensemble Models
- **Objective**: Improve detection rates using advanced models.
- **Methods**: Implemented Random Forest and Gradient Boosting classifiers.
- **Results**: Ensemble models outperformed the baseline with accuracy improvements of Z%.

## 4. Model Evaluation

### Experiment 7: Cross-Validation
- **Objective**: Validate model performance across different subsets of data.
- **Methods**: K-Fold cross-validation to assess model stability.
- **Results**: Consistent performance metrics across folds, indicating model reliability.

### Experiment 8: Evaluation Metrics
- **Objective**: Use appropriate metrics for imbalanced classification.
- **Methods**: Calculated precision, recall, F1-score, and ROC-AUC.
- **Results**: Identified the best model based on F1-score and ROC-AUC.

## 5. Model Explainability

### Experiment 9: SHAP Analysis
- **Objective**: Interpret model predictions and feature importance.
- **Methods**: Used SHAP values to analyze the contribution of each feature.
- **Results**: Provided insights into which features most influence fraud detection.

> Run the explainability script to regenerate SHAP artifacts locally:
>
> ```bash
> python scripts/explain_shap.py --data data/processed/Fraud_Data_processed.csv --output outputs/explainability
> ```
> This saves global and per-instance SHAP visualizations and a short `docs/TASK_3_SHAP.md` report.

### Experiment 10: LIME Analysis
- **Objective**: Generate local explanations for individual predictions.
- **Methods**: Implemented LIME to explain model predictions on sample transactions.
- **Results**: Enhanced understanding of model behavior and decision-making process.

## Conclusion

The experiments conducted in this project have significantly contributed to improving the detection of fraudulent transactions. The combination of thorough data analysis, effective feature engineering, robust model building, and explainability techniques has led to a more reliable and interpretable fraud detection system. Future work will focus on further refining models and exploring additional data sources for enhanced performance.