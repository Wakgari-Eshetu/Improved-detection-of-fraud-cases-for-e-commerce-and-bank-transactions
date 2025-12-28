def calculate_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def calculate_precision(y_true, y_pred):
    true_positive = ((y_true == 1) & (y_pred == 1)).sum()
    false_positive = ((y_true == 0) & (y_pred == 1)).sum()
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0

def calculate_recall(y_true, y_pred):
    true_positive = ((y_true == 1) & (y_pred == 1)).sum()
    false_negative = ((y_true == 1) & (y_pred == 0)).sum()
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

def calculate_f1_score(y_true, y_pred):
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def calculate_roc_auc(y_true, y_scores):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_scores)

def calculate_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred)