def test_no_duplicates(sample_data):
    df = sample_data
    df_cleaned = df.drop_duplicates()
    assert df_cleaned.duplicated().sum() == 0

def test_data_types(sample_data):
    df = sample_data
    # Convert categorical columns
    cat_cols = ['device_id','source','browser','sex']
    df[cat_cols] = df[cat_cols].astype('category')
    for col in cat_cols:
        assert df[col].dtype.name == 'category'
