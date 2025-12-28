# src/data/load_data.py
import os
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    print(f"[INFO] Loaded {file_path} with shape {data.shape}")
    return data

def load_raw_data(directory: str) -> pd.DataFrame:
  
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = load_data(file_path)
            all_data.append(df)
            
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"[INFO] Combined dataset shape: {combined_data.shape}")
        return combined_data
    else:
        print("[WARNING] No CSV files found in the directory.")
        return pd.DataFrame()
