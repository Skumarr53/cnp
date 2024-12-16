# nli_finetune/metrics.py
import time
import evaluate
import numpy as np
import pandas as pd
import mlflow
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
    
    pred_labels = (preds > entailment_threshold).astype(int)


    accuracy = accuracy_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels, average='binary', zero_division=0)
    recall = recall_score(labels, pred_labels, average='binary', zero_division=0)
    f1 = f1_score(labels, pred_labels, average='binary', zero_division=0)
    
    # Assuming binary classification
    try:
        roc_auc = roc_auc_score(labels, preds)
    except ValueError:
        roc_auc = float('nan')


    return {
        "predictions": pred_labels,
        "scores": preds,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

def get_compute_metrics(
    is_regression: bool,
    task_name: Optional[str] = None,
    eval_entailment_thresold: float = 0.5,
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

        return compute_metrics(probabilities, p.label_ids, eval_entailment_thresold)


    return get_metrics

def _evaluate_model(
    run_id: str,
    run_name: str,
    model,
    tokenizer,
    model_family_name: str,
    num_train_epochs: Any,
    learning_rate: Any,
    weight_decay: Any,
    train_batch_size: Any,
    eval_batch_size: Any,
    eval_df: pd.DataFrame,
    entailment_threshold: float
) -> Dict[str, Any]:
    """
    Evaluate a single NLI model and compile the results.

    Args:
        run_id (str): MLflow run ID or model identifier.
        run_name (str): Name of the run.
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        model_family_name (str): Name of the model family.
        num_train_epochs (Any): Number of training epochs.
        learning_rate (Any): Learning rate used during training.
        weight_decay (Any): Weight decay used during training.
        train_batch_size (Any): Training batch size.
        eval_batch_size (Any): Evaluation batch size.
        eval_df (pd.DataFrame): Evaluation dataset.
        entailment_threshold (float): Threshold to determine entailment.

    Returns:
        Dict[str, Any]: Dictionary containing the evaluation results for the model.
    """
    try:
        logger.info(f"Starting evaluation for Run ID={run_id}, Run Name={run_name}")

        # Initialize the pipeline
        nli_pipeline = pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Start timing
        start_time = time.time()

        # Compute metrics
        metrics = get_nli_model_metrics(nli_pipeline, eval_df, entailment_threshold)

        # End timing
        time_taken = time.time() - start_time

        # Compile results
        result = ModelEvaluationResult(
            run_id=run_id,
            run_name=run_name,
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

        logger.info(f"Completed evaluation for Run ID={run_id}")
        return asdict(result)

    except Exception as e:
        logger.error(f"Error evaluating model Run ID={run_id}: {e}")
        return {}

def evaluate_nli_models_from_path(
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
        logger.info(f"Evaluating model at path: {model_path}")

        try:
            # Load the model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)

            # Extract fine-tuning parameters from the configuration or a separate file
            config = model.config
            model_family_name = getattr(config, 'model_family', 'Unknown')
            num_train_epochs = getattr(config, 'num_train_epochs', 'Unknown')
            learning_rate = getattr(config, 'learning_rate', 'Unknown')
            weight_decay = getattr(config, 'weight_decay', 'Unknown')
            train_batch_size = getattr(config, 'train_batch_size', 'Unknown')
            eval_batch_size = getattr(config, 'eval_batch_size', 'Unknown')

            # Use the helper function to evaluate the model
            # Assuming run_id and run_name are not applicable for local models
            result = _evaluate_model(
                run_id=model_path,  # Using model path as run_id
                run_name='Local Model',
                model=model,
                tokenizer=tokenizer,
                model_family_name=model_family_name,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                eval_df=eval_df,
                entailment_threshold=entailment_threshold
            )

            if result:
                results.append(result)

        except Exception as e:
            logger.error(f"Failed to evaluate model at path {model_path}: {e}")
            continue  # Proceed to the next model

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df



def evaluate_nli_models_mlflow(
    experiment_name: str,
    csv_path: str,
    metric: str = 'accuracy',
    entailment_threshold: float = 0.7
) -> pd.DataFrame:
    """
    Evaluate and compare NLI models logged in a specified MLflow experiment on a provided dataset.

    Args:
        experiment_name (str): Name of the MLflow experiment containing the models.
        csv_path (str): Path to the CSV dataset for evaluation.
        metric (str, optional): Metric to sort models by when listing. Default is 'accuracy'.
        entailment_threshold (float, optional): Threshold to determine entailment. Default is 0.7.

    Returns:
        pd.DataFrame: DataFrame containing evaluation results for each model.
    """
    results = []

    # Load the dataset
    eval_df = pd.read_csv(csv_path)

    # Prepare ground truth labels
    eval_df['label_GT'] = eval_df['label'].apply(lambda x: 1 if x == 'entailment' else 0)

    # Retrieve available models from MLflow experiment
    available_models = list_available_models(experiment_name, metric=metric)
    if not available_models:
        logger.warning("No models found to evaluate.")
        return pd.DataFrame()  # Return empty DataFrame if no models are found

    for model_info in available_models:
        run_id = model_info['run_id']
        run_name = model_info['run_name']
        logger.info(f"Evaluating model: Run ID={run_id}, Run Name={run_name}")

        try:
            # Load the model from MLflow
            production_model = mlflow.transformers.load_model(run_id)
            model = production_model.model
            tokenizer = production_model.tokenizer

            # Extract fine-tuning parameters from MLflow run
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            params = run.data.params
            tags = run.data.tags

            # Assuming fine-tuning parameters are logged as parameters in MLflow
            num_train_epochs = float(params.get('num_train_epochs', 'nan'))
            learning_rate = float(params.get('learning_rate', 'nan'))
            weight_decay = float(params.get('weight_decay', 'nan'))
            train_batch_size = int(params.get('train_batch_size', 'nan'))
            eval_batch_size = int(params.get('eval_batch_size', 'nan'))

            # Optionally, extract model family name from tags or params
            model_family_name = tags.get('base_model', 'Unknown')

            # Use the helper function to evaluate the model
            result = _evaluate_model(
                run_id=run_id,
                run_name=run_name,
                model=model,
                tokenizer=tokenizer,
                model_family_name=model_family_name,
                num_train_epochs=num_train_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                eval_df=eval_df,
                entailment_threshold=entailment_threshold
            )

            if result:
                results.append(result)

        except Exception as e:
            logger.error(f"Failed to evaluate model with Run ID={run_id}: {e}")
            continue  # Proceed to the next model

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df