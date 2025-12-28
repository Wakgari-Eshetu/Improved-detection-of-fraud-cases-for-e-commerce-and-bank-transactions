# src/preprocessing/transform.py

import pandas as pd
import numpy as np
import ipaddress
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def ip_to_int(ip):
 
    try:
        return int(ipaddress.ip_address(ip))
    except Exception:
        return np.nan


def merge_ip_country(df, ip_df):

    df = df.copy()
    ip_df = ip_df.copy()

    # Convert fraud IP to integer
    df['ip_int'] = df['ip_address'].apply(ip_to_int)

    # Convert IP range bounds to integer
    ip_df['ip_start'] = ip_df['lower_bound_ip_address'].apply(ip_to_int)
    ip_df['ip_end'] = ip_df['upper_bound_ip_address'].apply(ip_to_int)

    # Ensure correct ordering
    ip_df = ip_df.sort_values('ip_start')

    # Assign country using range lookup
    def find_country(ip):
        if pd.isna(ip):
            return np.nan
        match = ip_df[(ip_df['ip_start'] <= ip) & (ip_df['ip_end'] >= ip)]
        return match.iloc[0]['country'] if not match.empty else np.nan

    df['country'] = df['ip_int'].apply(find_country)

    return df


def create_time_features(df):
    df = df.copy()

    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['signup_hour'] = df['signup_time'].dt.hour
        df['signup_day'] = df['signup_time'].dt.day
        df['signup_weekday'] = df['signup_time'].dt.weekday

    if 'purchase_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        df['purchase_hour'] = df['purchase_time'].dt.hour
        df['purchase_day'] = df['purchase_time'].dt.day
        df['purchase_weekday'] = df['purchase_time'].dt.weekday

    return df


def scale_and_encode(df, *args, **kwargs):
    df = df.copy()

    # Drop high-cardinality / raw columns
    drop_cols = ['ip_address', 'ip_int']
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    return df

