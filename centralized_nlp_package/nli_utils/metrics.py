# nli_finetune/metrics.py
import time
import evaluate
import numpy as np
import pandas as pd
from loguru import logger
from typing import List, Dict, Any, Callable, Optional

import torch
from transformers import (EvalPrediction,pipeline, Pipeline, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from dataclasses import dataclass, asdict

@dataclass
class ModelEvaluationResult:
    model_family_name: str
    entailment_threshold: float
    time_taken_seconds: float
    num_train_epochs: int
    learning_rate: float
    weight_decay: float
    train_batch_size: int
    eval_batch_size: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float


def get_nli_model_metrics(nli_pipeline: Pipeline, eval_df: pd.DataFrame, entailment_threshold: float = 0.5) -> ModelEvaluationResult:

    # Compute entailment scores
    preds = eval_df.apply(
        lambda x: nli_pipeline(
            sequences=x['sentence1'],
            candidate_labels=[x['sentence2']],
            hypothesis_template="{}",
            multi_label=True
        )['scores'][0], axis=1
    )
    print("eval data samples:", eval_df.head())
    eval_df['label_GT'] = eval_df['label'].apply(lambda x: 1 if x == 'entailment' else 0)
    labels = eval_df['label_GT'].values


    # Compute metrics
    metrics = compute_metrics(preds, labels,entailment_threshold = entailment_threshold)

    return metrics


def compute_metrics(preds: np.ndarray, labels: np.ndarray, entailment_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute additional metrics beyond accuracy.

    Args:
        preds (np.ndarray): Predicted labels.
        labels (np.ndarray): True labels.

    Returns:
        Dict[str, Any]: Dictionary of computed metrics.
    """
    
    print('Entailment scores with threshold: {}'.format(entailment_threshold),preds)
    pred_labels = (preds > entailment_threshold).astype(int)
    print("predicted labels", pred_labels)
    print("true labels", labels)


    accuracy = accuracy_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels, average='binary', zero_division=0)
    recall = recall_score(labels, pred_labels, average='binary', zero_division=0)
    f1 = f1_score(labels, pred_labels, average='binary', zero_division=0)
    
    # Assuming binary classification
    try:
        # If you have probability scores, use them for ROC-AUC
        # Here, preds are binary labels; you need the probabilities
        # This function should be modified accordingly if probabilities are available
        # For demonstration, we'll set ROC-AUC as nan
        roc_auc = roc_auc_score(labels, preds)
    except ValueError:
        roc_auc = float('nan')


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

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
    # if task_name is not None:
    #     metric = evaluate.load("glue", task_name)
    # elif is_regression:
    #     metric = evaluate.load("mse")
    # else:
    #     metric = evaluate.load("accuracy.py")

    def get_metrics(p: EvalPrediction) -> dict:

        print("predictions type", type(p.predictions))
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        # preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

        ## Softmax to convert logits to probabilities
        def softmax(logits):
            exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
            return exp_logits / exp_logits.sum(axis=0)

        probabilities = np.array([softmax(logit)[1] for logit in preds])

        return compute_metrics(probabilities, p.label_ids, entailment_threshold = 0.5)


    return get_metrics



def evaluate_nli_models(
    model_paths: List[str],
    csv_path: str,
    entailment_threshold: float = 0.7
) -> pd.DataFrame:
    """
    Evaluate and compare a list of fine-tuned NLI models on a provided dataset.

    Args:
        model_paths (List[str]): List of paths to fine-tuned models.
        csv_path (str): Path to the CSV dataset.
        entailment_threshold (float): Threshold to determine entailment.

    Returns:
        pd.DataFrame: DataFrame containing evaluation results for each model.
    """
    results = []

    # Load the dataset
    eval_df = pd.read_csv(csv_path)

    

    # Prepare ground truth labels
    eval_df['label_GT'] = eval_df['label'].apply(lambda x: 1 if x == 'entailment' else 0)

    for model_path in model_paths:
        print("Evaluating model:", model_path)
        # Start timing predictions
        start_time = time.time()

        # Load the model pipeline


        # Extract model family name from the model path or configuration
        # config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Retrieve fine-tuning parameters from the configuration or a separate file
        # This assumes that fine-tuning parameters are stored in the model's config
        # Modify this part if your fine-tuning parameters are stored differently
        model_family_name = model.config.model_family if hasattr(model.config, 'model_family') else None
        num_train_epochs = model.config.num_train_epochs if hasattr(model.config, 'num_train_epochs') else None
        learning_rate = model.config.learning_rate if hasattr(model.config, 'learning_rate') else None
        weight_decay = model.config.weight_decay if hasattr(model.config, 'weight_decay') else None
        train_batch_size = model.config.train_batch_size if hasattr(model.config, 'train_batch_size') else None
        eval_batch_size = model.config.eval_batch_size if hasattr(model.config, 'eval_batch_size') else None
        
        nli_pipeline = pipeline(
            "zero-shot-classification",
            model=model, tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1 )
        print("loading model:", model_path)
    
        ## get metrics 
        metrics = get_nli_model_metrics(nli_pipeline, eval_df, entailment_threshold)

        # End timing
        end_time = time.time()
        time_taken = end_time - start_time

        # Compile results
        result = ModelEvaluationResult(
            model_family_name=model_family_name,
            entailment_threshold=entailment_threshold,
            time_taken_seconds=time_taken,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            accuracy=metrics.get("accuracy", float('nan')),
            precision=metrics.get("precision", float('nan')),
            recall=metrics.get("recall", float('nan')),
            f1_score=metrics.get("f1_score", float('nan')),
            roc_auc=metrics.get("roc_auc", float('nan'))
        )

        results.append(asdict(result))


    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df