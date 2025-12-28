import pytest
from src.models.trainer import train_model
from src.models.pipelines import create_pipeline
from src.evaluation.metrics import calculate_metrics
from src.data.sampling import handle_class_imbalance

def test_train_model():
    # Mock data for testing
    X_train = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]
    y_train = [0, 1, 0]
    
    model = train_model(X_train, y_train)
    
    assert model is not None
    assert hasattr(model, 'predict')

def test_create_pipeline():
    pipeline = create_pipeline()
    
    assert pipeline is not None
    assert 'classifier' in pipeline.named_steps

def test_calculate_metrics():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 0]
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert 'accuracy' in metrics
    assert 'f1_score' in metrics

def test_handle_class_imbalance():
    y = [0] * 90 + [1] * 10  # Imbalanced dataset
    X_resampled, y_resampled = handle_class_imbalance(X, y)
    
    assert len(y_resampled) == 100
    assert sum(y_resampled) > 10  # Check if minority class is represented