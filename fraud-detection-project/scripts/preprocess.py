import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.preprocessing.clean import drop_duplicates, correct_dtypes
from src.preprocessing.impute import impute_missing
from src.preprocessing.transform import merge_ip_country, create_time_features, scale_and_encode
from src.data.sampling import handle_class_imbalance

def preprocess_fraud():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_folder = os.path.join(project_root, "data", "raw")
    processed_folder = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_folder, exist_ok=True)

    fraud_file = os.path.join(raw_folder, "Fraud_Data.csv")
    print("[INFO] Loading Fraud_Data.csv...")
    df = load_data(fraud_file)
    print(f"[INFO] Raw fraud data shape: {df.shape}")

    # Cleaning
    df = drop_duplicates(df)
    df = correct_dtypes(df)

    # Imputation
    df = impute_missing(df)

    ip_file = os.path.join(processed_folder, "IpAddress_to_Country_processed.csv")
    ip_df = load_data(ip_file)

    # Feature transformations
    df = merge_ip_country(df,ip_df)
    df = create_time_features(df)
    df = scale_and_encode(df)

    # Save processed
    out_file = os.path.join(processed_folder, "Fraud_Data_processed.csv")
    df.to_csv(out_file, index=False)
    print(f"[INFO] Processed fraud data saved: {out_file}")

    return df

def preprocess_creditcard():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_folder = os.path.join(project_root, "data", "raw")
    processed_folder = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_folder, exist_ok=True)

    credit_file = os.path.join(raw_folder, "creditcard.csv")
    print("[INFO] Loading creditcard.csv...")
    df = load_data(credit_file)
    print(f"[INFO] Raw creditcard data shape: {df.shape}")

    # Cleaning
    df = drop_duplicates(df)
    df = correct_dtypes(df)

    # Imputation
    df = impute_missing(df)
    num_features = df.select_dtypes(include=['int64', 'float64']).columns
    cat_features = df.select_dtypes(include=['object', 'category']).columns

    # Feature transformations
    df = scale_and_encode(df,numeric_features=num_features,categorical_features=cat_features)  # no IP merge for creditcard

    # Save processed
    out_file = os.path.join(processed_folder, "creditcard_processed.csv")
    df.to_csv(out_file, index=False)
    print(f"[INFO] Processed creditcard data saved: {out_file}")

    return df

def preprocess_ip_country():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_folder = os.path.join(project_root, "data", "raw")
    processed_folder = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_folder, exist_ok=True)

    ip_file = os.path.join(raw_folder, "IpAddress_to_Country.csv")
    print("[INFO] Loading IpAddress_to_Country.csv...")
    df = load_data(ip_file)
    print(f"[INFO] Raw IP-country data shape: {df.shape}")

    # Only cleaning needed
    df = drop_duplicates(df)
    df = correct_dtypes(df)

    out_file = os.path.join(processed_folder, "IpAddress_to_Country_processed.csv")
    df.to_csv(out_file, index=False)
    print(f"[INFO] Processed IP-country data saved: {out_file}")

    return df

if __name__ == "__main__":
    ip_df = preprocess_ip_country()
    fraud_df = preprocess_fraud()
    credit_df = preprocess_creditcard()
    
    print("[INFO] All datasets preprocessed successfully!")


