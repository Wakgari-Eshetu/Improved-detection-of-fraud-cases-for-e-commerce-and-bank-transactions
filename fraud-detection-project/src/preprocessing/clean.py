def clean_data(df):
   
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Correct data types (example)
    # df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    # df['amount'] = df['amount'].astype(float)
    
    # Handle missing values (example)
    # df['amount'].fillna(df['amount'].mean(), inplace=True)
    
    return df