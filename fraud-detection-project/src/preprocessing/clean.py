# src/preprocessing/clean.py
import pandas as pd

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    print(f"[INFO] Duplicates removed. New shape: {df.shape}")
    return df

def correct_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # Convert timestamps to datetime
    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
    if 'purchase_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df
