from sklearn.metrics import accuracy_score
import numpy as np 

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tabulate import tabulate
import json

def compute_metrics_for_pair(eval_preds):
    predictions = eval_preds.predictions  # logits (or whatever Trainer returned)
    labels = eval_preds.label_ids          # true labels
    preds = np.argmax(predictions, axis=1).reshape(-1)

    metric = {
        "accuracy": float(accuracy_score(labels, preds))
    }
    return metric

def compute_generation_metrics(predictions, references, contexts):
    """Compute BLEU and ROUGE-L for each prediction-reference pair."""
    smoother = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    assert len(predictions) == len(references) == len(contexts), f'unequal lengths -- context: {len(contexts)} -- references: {len(references)} -- predictions: {len(predictions)}'

    results = []
    for pred, ref, context in zip(predictions, references, contexts):
        # BLEU
        bleu = sentence_bleu(
            [ref.split()],
            pred.split(),
            smoothing_function=smoother
        )

        # ROUGE
        rouge = scorer.score(ref, pred)
        rouge1 = rouge['rouge1'].fmeasure
        rougeL = rouge['rougeL'].fmeasure

        results.append({
            "context":context,
            "reference": ref,
            "prediction": pred,
            "bleu": bleu,
            "rouge1": rouge1,
            "rougeL": rougeL
        })
    return results
