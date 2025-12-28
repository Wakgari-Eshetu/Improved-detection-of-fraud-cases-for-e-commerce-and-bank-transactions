from sklearn.model_selection import StratifiedKFold
import numpy as np

def cross_validate(model, X, y, n_splits=5, random_state=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)

    return np.array(scores)