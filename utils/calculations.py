from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(y_pred):
    y_true = [1] * len(y_pred)  
    return {
        'accuracy': y_pred.mean(),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'valid_pairs': y_pred.sum(),
        'total_pairs': len(y_pred)
    }