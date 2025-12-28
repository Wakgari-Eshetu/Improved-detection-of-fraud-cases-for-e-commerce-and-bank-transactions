from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_model(model_name, **kwargs):
    if model_name == 'logistic_regression':
        return LogisticRegression(**kwargs)
    elif model_name == 'random_forest':
        return RandomForestClassifier(**kwargs)
    elif model_name == 'svm':
        return SVC(**kwargs)
    elif model_name == 'xgboost':
        return XGBClassifier(**kwargs)
    else:
        raise ValueError(f"Model {model_name} is not recognized.")