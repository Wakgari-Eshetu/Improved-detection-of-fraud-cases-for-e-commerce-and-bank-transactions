import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    # Small sample data for testing
    data = {
        'user_id': [1, 1, 2, 3],
        'signup_time': pd.to_datetime(['2023-01-01','2023-01-01','2023-01-02','2023-01-03']),
        'purchase_time': pd.to_datetime(['2023-01-01 10:00','2023-01-01 15:00','2023-01-02 11:00','2023-01-03 12:00']),
        'purchase_value': [100, 150, 200, 50],
        'device_id': ['d1','d1','d2','d3'],
        'source': ['web','web','app','app'],
        'browser': ['chrome','chrome','safari','firefox'],
        'sex': ['M','M','F','F'],
        'age': [25,25,30,22],
        'ip_address': ['192.168.0.1','192.168.0.1','10.0.0.1','10.0.0.2'],
        'class': [0,0,1,0]
    }
    df = pd.DataFrame(data)
    return df
