from sklearn.metrics import accuracy_score
import numpy as np 

def compute_metrics_for_pair(eval_preds):
    predictions = eval_preds.predictions  # logits (or whatever Trainer returned)
    labels = eval_preds.label_ids          # true labels
    preds = np.argmax(predictions, axis=1).reshape(-1)

    metric = {
        "accuracy": float(accuracy_score(labels, preds))
    }
    return metric