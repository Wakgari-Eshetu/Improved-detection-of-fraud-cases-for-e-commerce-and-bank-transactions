# src/preprocessing/impute.py
import pandas as pd

def impute_missing(df: pd.DataFrame, strategy='mean', columns=None) -> pd.DataFrame:
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()
    
    for col in columns:
        if strategy == 'mean':
            df[col].fillna(df[col].mean(), inplace=True)
        elif strategy == 'median':
            df[col].fillna(df[col].median(), inplace=True)
        elif strategy == 'mode':
            df[col].fillna(df[col].mode()[0], inplace=True)
    print("[INFO] Missing values imputed.")
    return df
