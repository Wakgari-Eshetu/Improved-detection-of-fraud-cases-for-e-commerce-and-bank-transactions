from imblearn.over_sampling import SMOTE

def test_smote_resampling(sample_data):
    df = sample_data.copy()
    X = df[['purchase_value','age']]
    y = df['class']
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Check that classes are balanced
    class_counts = y_res.value_counts()
    assert class_counts[0] == class_counts[1]
