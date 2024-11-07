# nli_finetune/metrics.py

import evaluate
import numpy as np
from transformers import EvalPrediction
from typing import Optional, Callable

def get_compute_metrics(
    is_regression: bool,
    task_name: Optional[str] = None
) -> Callable[[EvalPrediction], dict]:
    """
    Returns a metrics computation function based on the task.

    Args:
        is_regression (bool): Whether the task is a regression task.
        task_name (Optional[str]): The name of the GLUE task.

    Returns:
        Callable[[EvalPrediction], dict]: A function that computes metrics.

    Usage Example:
        >>> from transformers import EvalPrediction
        >>> compute_metrics = get_compute_metrics(is_regression=False, task_name="mnli")
        >>> preds = np.array([[0.1, 0.9], [0.8, 0.2]])
        >>> labels = np.array([1, 0])
        >>> eval_pred = EvalPrediction(predictions=preds, label_ids=labels)
        >>> metrics = compute_metrics(eval_pred)
    """
    if task_name is not None:
        metric = evaluate.load("glue", task_name)
    elif is_regression:
        metric = evaluate.load("mse")
    else:
        metric = evaluate.load("accuracy")

    def compute_metrics(p: EvalPrediction) -> dict:
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    return compute_metrics
