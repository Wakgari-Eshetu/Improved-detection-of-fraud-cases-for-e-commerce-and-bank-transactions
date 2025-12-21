Fraud Detection for E-commerce and Bank Transactions
Project Overview

This project focuses on detecting fraudulent transactions in e-commerce and bank datasets. The dataset contains both numerical and categorical features, including user information, transaction details, and device/browser metadata.

The project workflow includes data cleaning, exploratory data analysis (EDA), feature engineering, data transformation, and handling class imbalance, leading up to preparing data for predictive modeling.

Task 1: Data Preparation & Cleaning
1. Data Cleaning

Remove duplicates: Ensured each transaction is unique.

Correct data types:

Categorical features: source, browser, sex, device_id, country.

Numerical features: purchase_value, age, user_transaction_count, etc.

Handle missing values:

Numerical: Imputed with median.

Categorical: Imputed with most frequent value.

2. Check for issues

Checked for duplicate rows and removed them.

Checked for missing values and handled appropriately.

Task 2: Exploratory Data Analysis (EDA)
Univariate Analysis

Distribution of key numerical features: purchase_value, age, transactions_last_24h.

Distribution of categorical features: source, browser, sex, country.

Bivariate Analysis

Relationship between features and target (class).

Boxplots for transaction amount vs fraud class.

Visualized time_since_last_transaction and time_since_signup by fraud class.

Class Distribution Analysis

Quantified imbalance between fraud (1) and non-fraud (0) classes.

Task 3: Geolocation Integration

Converted ip_address to integer format (ip_int).

Merged Fraud_Data.csv with IpAddress_to_Country.csv using range-based lookup.

Analyzed fraud patterns by country, including "Unknown" where IP range could not be matched.

Task 4: Feature Engineering

Transaction frequency and velocity: Number of transactions per user in rolling time windows (e.g., last 24 hours).

Time-based features:

hour_of_day

day_of_week

time_since_signup

time_since_last_transaction

Task 5: Data Transformation

Scale numerical features:

Used StandardScaler for features like purchase_value, age, transactions_last_24h, etc.

Encode categorical features:

One-Hot Encoding for device_id, source, browser, sex.

Used sparse matrices to handle large dimensionality efficiently.

Task 6: Handle Class Imbalance

Problem: Fraud class is heavily underrepresented.

Solution: Applied SMOTE (Synthetic Minority Oversampling Technique) on the training set only.

Avoided altering test set to ensure realistic evaluation.

Justification: SMOTE generates synthetic minority samples, preventing overfitting and allowing the model to learn better from minority class.

Class Distribution

Before SMOTE: Fraud cases are ~4% of total transactions.

After SMOTE: Fraud and non-fraud classes are balanced.

