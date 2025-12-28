# src/features/build_features.py

import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE


def make_data_ready_for_training(
    data_path: str,
    target_col: str,
    drop_cols: list | None = None,
    output_name: str = "Fraud_Data_ready.csv"
):
    # -------------------------------
    # Load processed data
    # -------------------------------
    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded processed data: {df.shape}")

    # -------------------------------
    # Drop unnecessary columns
    # -------------------------------
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # -------------------------------
    # Split X and y
    # -------------------------------
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # -------------------------------
    # Identify numeric & categorical features
    # -------------------------------
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # -------------------------------
    # Preprocessing (scale + encode)
    # -------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), cat_features),
        ]
    )

    X_transformed = preprocessor.fit_transform(X)

    # -------------------------------
    # Get feature names after encoding
    # -------------------------------
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features)
    feature_names = num_features + cat_feature_names.tolist()

    X_encoded = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

    # -------------------------------
    # Handle missing values before SMOTE
    # -------------------------------
    num_cols_encoded = X_encoded.select_dtypes(include=["float64", "int64"]).columns
    cat_cols_encoded = X_encoded.select_dtypes(exclude=["float64", "int64"]).columns

    if num_features:
        num_imputer = SimpleImputer(strategy="median")
        X_num_imputed = pd.DataFrame(
            num_imputer.fit_transform(X[num_features]),
            columns=num_features,
            index=X.index
        )
        X[num_features] = X_num_imputed



    if cat_features:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_cat_imputed = pd.DataFrame(
            cat_imputer.fit_transform(X[cat_features]),
            columns=cat_features,
            index=X.index
        )
        X[cat_features] = X_cat_imputed


    # -------------------------------
    # Handle class imbalance with SMOTE
    # -------------------------------
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

    # -------------------------------
    # Save ready-for-training data
    # -------------------------------
    ready_df = X_resampled.copy()
    ready_df[target_col] = y_resampled.values

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ready_folder = os.path.join(project_root, "data", "readyfortraining")
    os.makedirs(ready_folder, exist_ok=True)

    output_path = os.path.join(ready_folder, output_name)
    ready_df.to_csv(output_path, index=False)

    print(f"[INFO] Training-ready data saved to: {output_path}")
    print(f"[INFO] Final shape: {ready_df.shape}")

    return X_resampled, y_resampled


# -------------------------------
# Run the function
# -------------------------------
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    processed_file = os.path.join(project_root, "data", "processed", "Fraud_Data_processed.csv")

    make_data_ready_for_training(
        data_path=processed_file,
        target_col="class",
        drop_cols=["user_id", "signup_time", "purchase_time"]
    )
