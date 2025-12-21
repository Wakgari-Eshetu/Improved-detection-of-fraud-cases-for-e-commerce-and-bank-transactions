def test_time_based_features(sample_data):
    df = sample_data.copy()
    df = df.sort_values(['user_id','purchase_time'])
    # time_since_signup
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()/60
    assert 'time_since_signup' in df.columns
    assert all(df['time_since_signup'] >= 0)
    
    # hour_of_day
    df['hour_of_day'] = df['purchase_time'].dt.hour
    assert df['hour_of_day'].between(0,23).all()
    
    # day_of_week
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    assert df['day_of_week'].between(0,6).all()
