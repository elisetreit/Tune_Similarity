def calculate_accuracy(y_true, y_pred):
    """Calculate the accuracy of predictions."""
    correct_predictions = (y_true == y_pred).sum()
    total_predictions = len(y_true)
    return correct_predictions / total_predictions if total_predictions > 0 else 0.0

def calculate_precision(y_true, y_pred):
    """Calculate the precision of predictions."""
    true_positives = ((y_true == 1) & (y_pred == 1)).sum()
    predicted_positives = (y_pred == 1).sum()
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0

def calculate_recall(y_true, y_pred):
    """Calculate the recall of predictions."""
    true_positives = ((y_true == 1) & (y_pred == 1)).sum()
    actual_positives = (y_true == 1).sum()
    return true_positives / actual_positives if actual_positives > 0 else 0.0

def calculate_f1_score(y_true, y_pred):
    """Calculate the F1 score of predictions."""
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0