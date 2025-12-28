def create_features(df):
    # Example feature engineering
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    df['hour'] = df['transaction_time'].dt.hour
    df['day_of_week'] = df['transaction_time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Creating transaction frequency features
    df['transaction_count'] = df.groupby('user_id')['transaction_id'].transform('count')
    
    # Handling class imbalance by creating a new feature
    df['is_fraud'] = df['label'].apply(lambda x: 1 if x == 'fraud' else 0)
    
    return df

def feature_selection(df):
    # Selecting relevant features for modeling
    features = ['hour', 'is_weekend', 'transaction_count']
    target = 'is_fraud'
    
    return df[features], df[target]