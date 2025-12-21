from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import sparse

def test_numerical_scaling(sample_data):
    df = sample_data.copy()
    num_features = ['purchase_value','age']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_features])
    # Check mean ~0 and std ~1
    np.testing.assert_almost_equal(X_scaled.mean(axis=0), [0,0], decimal=1)
    np.testing.assert_almost_equal(X_scaled.std(axis=0), [1,1], decimal=1)

def test_categorical_encoding(sample_data):
    df = sample_data.copy()
    cat_features = ['device_id','source','browser','sex']
    encoder = OneHotEncoder(sparse_output=True)
    X_cat = encoder.fit_transform(df[cat_features])
    assert sparse.issparse(X_cat)
