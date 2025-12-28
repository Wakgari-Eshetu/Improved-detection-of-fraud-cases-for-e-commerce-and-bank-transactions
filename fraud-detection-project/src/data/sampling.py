#src/data/sampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

def handle_class_imbalance(X, y, method='smote', sampling_strategy='auto'):
    if method == 'smote':
        smote = SMOTE(sampling_strategy=sampling_strategy)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    elif method == 'undersample':
        undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy)
        X_resampled, y_resampled = undersampler.fit_resample(X, y)
    else:
        raise ValueError("Method must be 'smote' or 'undersample'.")

    print(f"Original dataset shape: {Counter(y)}")
    print(f"Resampled dataset shape: {Counter(y_resampled)}")

    return X_resampled, y_resampled