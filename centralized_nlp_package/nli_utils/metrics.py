# nli_finetune/metrics.py
import time
import evaluate
import numpy as np
import pandas as pd
import mlflow
#from loguru import logger
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
    """
    Compute evaluation metrics for an NLI model using a provided pipeline and dataset.

    This function applies a zero-shot classification pipeline to each example in the evaluation DataFrame to compute entailment scores. It then calculates various performance metrics based on these predictions and returns the results encapsulated in a `ModelEvaluationResult` dataclass.

    Args:
        nli_pipeline (Pipeline):
            A Hugging Face Transformers pipeline configured for zero-shot classification.
        eval_df (pd.DataFrame):
            A Pandas DataFrame containing the evaluation dataset. It must include the columns 'sentence1', 'sentence2', and 'label'.
        entailment_threshold (float, optional):
            The threshold above which a prediction is considered an entailment. Defaults to 0.5.

    Returns:
        ModelEvaluationResult:
            A dataclass instance containing the computed evaluation metrics and relevant training parameters.

    Raises:
        KeyError:
            If the required columns ('sentence1', 'sentence2', 'label') are not present in `eval_df`.
        Exception:
            If there is an error during the metric computation process.

    Example:
        >>> from transformers import pipeline
        >>> import pandas as pd
        >>> nli_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        >>> eval_data = pd.DataFrame({
        ...     'sentence1': ["The sky is blue.", "Cats are animals."],
        ...     'sentence2': ["It is sunny today.", "Dogs are mammals."],
        ...     'label': ["entailment", "entailment"]
        ... })
        >>> metrics = get_nli_model_metrics(nli_pipeline, eval_data, entailment_threshold=0.7)
        >>> print(metrics)
        ModelEvaluationResult(model_family_name='Unknown', entailment_threshold=0.7, time_taken_seconds=0.123, num_train_epochs=0, learning_rate=0.0, weight_decay=0.0, train_batch_size=0, eval_batch_size=0, accuracy=1.0, precision=1.0, recall=1.0, f1_score=1.0, roc_auc=1.0)
    """
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
    Compute standard classification metrics based on predictions and true labels.

    This function converts raw prediction scores into binary labels using a specified entailment threshold and then computes various performance metrics such as accuracy, precision, recall, F1 score, and ROC AUC.

    Args:
        preds (np.ndarray):
            An array of predicted entailment scores for each example.
        labels (np.ndarray):
            An array of true binary labels (1 for entailment, 0 otherwise).
        entailment_threshold (float, optional):
            The threshold above which a prediction is considered an entailment. Defaults to 0.5.

    Returns:
        Dict[str, Any]:
            A dictionary containing the following keys and their corresponding computed values:
                - "predictions" (np.ndarray): Binary predicted labels after applying the threshold.
                - "scores" (np.ndarray): Original entailment scores.
                - "accuracy" (float): Accuracy of the predictions.
                - "precision" (float): Precision of the predictions.
                - "recall" (float): Recall of the predictions.
                - "f1_score" (float): F1 score of the predictions.
                - "roc_auc" (float): ROC AUC score of the predictions.

    Raises:
        ValueError:
            If the input arrays `preds` and `labels` have mismatched lengths.
        Exception:
            If there is an error during metric computation.

    Example:
        >>> preds = np.array([0.6, 0.4, 0.8, 0.3])
        >>> labels = np.array([1, 0, 1, 0])
        >>> metrics = compute_metrics(preds, labels, entailment_threshold=0.5)
        >>> print(metrics)
        {'predictions': array([1, 0, 1, 0]), 'scores': array([0.6, 0.4, 0.8, 0.3]), 'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'roc_auc': 1.0}
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
    Generate a metrics computation function tailored to the task type.

    This function returns a callable that computes evaluation metrics based on whether the task is a regression or classification task. For NLI tasks, it processes the predictions to compute binary entailment metrics.

    Args:
        is_regression (bool):
            Indicates whether the task is a regression task. If `False`, it is treated as a classification task.
        task_name (Optional[str], optional):
            The name of the specific task (e.g., "mnli", "qqp"). If provided, it can be used to load task-specific metrics. Defaults to `None`.
        eval_entailment_thresold (float, optional):
            The threshold to determine entailment in binary classification tasks. Defaults to 0.5.

    Returns:
        Callable[[EvalPrediction], dict]:
            A function that takes an `EvalPrediction` object (containing predictions and true labels) and returns a dictionary of computed metrics.

    Example:
        >>> from transformers import EvalPrediction
        >>> compute_metrics_fn = get_compute_metrics(is_regression=False, eval_entailment_thresold=0.7)
        >>> preds = np.array([0.8, 0.4, 0.6])
        >>> labels = np.array([1, 0, 1])
        >>> eval_pred = EvalPrediction(predictions=preds, label_ids=labels)
        >>> metrics = compute_metrics_fn(eval_pred)
        >>> print(metrics)
        {'predictions': array([1, 0, 1]), 'scores': array([0.8, 0.4, 0.6]), 'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'roc_auc': 1.0}
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
    Evaluate a single NLI model and compile the evaluation results.

    This helper function initializes a zero-shot classification pipeline using the provided model and tokenizer, computes evaluation metrics on the given dataset, measures the time taken for evaluation, and returns the results in a structured dictionary.

    Args:
        run_id (str):
            The MLflow run ID or a unique identifier for the model being evaluated.
        run_name (str):
            The name of the run associated with the model evaluation.
        model (AutoModelForSequenceClassification):
            The trained NLI model to be evaluated.
        tokenizer (AutoTokenizer):
            The tokenizer corresponding to the trained model.
        model_family_name (str):
            The name of the model family (e.g., "DeBERTa", "FinBERT").
        num_train_epochs (Any):
            The number of training epochs the model underwent.
        learning_rate (Any):
            The learning rate used during training.
        weight_decay (Any):
            The weight decay (L2 regularization) applied during training.
        train_batch_size (Any):
            The batch size used for training.
        eval_batch_size (Any):
            The batch size used for evaluation.
        eval_df (pd.DataFrame):
            The evaluation dataset containing sentences and labels.
        entailment_threshold (float):
            The threshold to determine entailment in predictions.

    Returns:
        Dict[str, Any]:
            A dictionary containing the evaluation results, including metrics and training parameters.

    Raises:
        Exception:
            If there is an error during the evaluation process.

    Example:
        >>> from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        >>> tokenizer = AutoTokenizer.from_pretrained("deberta-base")
        >>> model = AutoModelForSequenceClassification.from_pretrained("deberta-base")
        >>> eval_data = pd.DataFrame({
        ...     'sentence1': ["The sky is blue.", "Cats are animals."],
        ...     'sentence2': ["It is sunny today.", "Dogs are mammals."],
        ...     'label': ["entailment", "entailment"]
        ... })
        >>> result = _evaluate_model(
        ...     run_id="12345",
        ...     run_name="deberta_run",
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     model_family_name="DeBERTa",
        ...     num_train_epochs=3,
        ...     learning_rate=2e-5,
        ...     weight_decay=0.01,
        ...     train_batch_size=16,
        ...     eval_batch_size=16,
        ...     eval_df=eval_data,
        ...     entailment_threshold=0.7
        ... )
        >>> print(result)
        {'model_family_name': 'DeBERTa', 'entailment_threshold': 0.7, 'time_taken_seconds': 0.456, 'num_train_epochs': 3, 'learning_rate': 2e-05, 'weight_decay': 0.01, 'train_batch_size': 16, 'eval_batch_size': 16, 'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'roc_auc': 1.0}
    """
    try:
        print("Starting evaluation for Run ID={run_id}, Run Name={run_name}")

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

        print("Completed evaluation for Run ID={run_id}")
        return asdict(result)

    except Exception as e:
        print("Error evaluating model Run ID={run_id}: {e}")
        return {}

def evaluate_nli_models_from_path(
    model_paths: List[str],
    csv_path: str,
    entailment_threshold: float = 0.7
) -> pd.DataFrame:
    """
    Evaluate and compare a list of fine-tuned NLI models on a provided dataset.

    This function iterates through a list of model paths, loads each model and its tokenizer, evaluates the model on the provided dataset, and compiles the evaluation results into a Pandas DataFrame.

    Args:
        model_paths (List[str]):
            A list of file paths to the fine-tuned NLI models to be evaluated.
        csv_path (str):
            Path to the CSV file containing the evaluation dataset. The CSV must include the columns 'sentence1', 'sentence2', and 'label'.
        entailment_threshold (float, optional):
            The threshold above which a prediction is considered an entailment. Defaults to 0.7.

    Returns:
        pd.DataFrame:
            A DataFrame containing evaluation results for each model, with columns corresponding to metrics and training parameters.

    Raises:
        FileNotFoundError:
            If the evaluation dataset CSV file does not exist.
        Exception:
            If there is an error during the evaluation process for any of the models.

    Example:
        >>> model_paths = ["models/deberta_finetuned", "models/finbert_finetuned"]
        >>> evaluation_results = evaluate_nli_models_from_path(model_paths, "data/eval_dataset.csv", entailment_threshold=0.8)
        >>> print(evaluation_results)
          model_family_name  entailment_threshold  time_taken_seconds  num_train_epochs  learning_rate  weight_decay  train_batch_size  eval_batch_size  accuracy  precision  recall  f1_score  roc_auc
        0          DeBERTa                   0.8                 120.5                 3         2e-05           0.01                16                16      0.85       0.80    0.75      0.77      0.83
        1           FinBERT                   0.8                 150.3                 4         3e-05           0.02                32                16      0.88       0.82    0.80      0.81      0.85
    """
    results = []

    # Load the dataset
    eval_df = pd.read_csv(csv_path)

    # Prepare ground truth labels
    eval_df['label_GT'] = eval_df['label'].apply(lambda x: 1 if x == 'entailment' else 0)

    for model_path in model_paths:
        print("Evaluating model at path: {model_path}")

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
            print("Failed to evaluate model at path {model_path}: {e}")
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

    This function retrieves all models from the specified MLflow experiment, loads each model along with its tokenizer, evaluates the model on the provided dataset, and compiles the evaluation results into a Pandas DataFrame.

    Args:
        experiment_name (str):
            The name of the MLflow experiment containing the models to be evaluated.
        csv_path (str):
            Path to the CSV file containing the evaluation dataset. The CSV must include the columns 'sentence1', 'sentence2', and 'label'.
        metric (str, optional):
            The metric used to sort and retrieve the best models from the experiment. Defaults to 'accuracy'.
        entailment_threshold (float, optional):
            The threshold above which a prediction is considered an entailment. Defaults to 0.7.

    Returns:
        pd.DataFrame:
            A DataFrame containing evaluation results for each model, with columns corresponding to metrics and training parameters.

    Raises:
        FileNotFoundError:
            If the evaluation dataset CSV file does not exist.
        mlflow.exceptions.MlflowException:
            If there is an issue accessing the MLflow experiment or runs.
        Exception:
            If there is an error during the evaluation process for any of the models.

    Example:
        >>> evaluation_results = evaluate_nli_models_mlflow(
        ...     experiment_name="NLI_Experiment",
        ...     csv_path="data/eval_dataset.csv",
        ...     metric="f1_score",
        ...     entailment_threshold=0.75
        ... )
        >>> print(evaluation_results)
          run_id           run_name          model_family_name  entailment_threshold  time_taken_seconds  num_train_epochs  learning_rate  weight_decay  train_batch_size  eval_batch_size  accuracy  precision  recall  f1_score  roc_auc
        0  12345abcdef  deberta_run_v1      DeBERTa                 0.75                 120.5                 3         2e-05           0.01                16                16      0.85       0.80    0.75      0.77      0.83
        1  67890ghijkl  finbert_run_v2       FinBERT                 0.75                 150.3                 4         3e-05           0.02                32                16      0.88       0.82    0.80      0.81      0.85
    """
    results = []

    # Load the dataset
    eval_df = pd.read_csv(csv_path)

    # Prepare ground truth labels
    eval_df['label_GT'] = eval_df['label'].apply(lambda x: 1 if x == 'entailment' else 0)

    # Retrieve available models from MLflow experiment
    available_models = list_available_models(experiment_name, metric=metric)
    if not available_models:
        print("No models found to evaluate.")
        return pd.DataFrame()  # Return empty DataFrame if no models are found

    for model_info in available_models:
        run_id = model_info['run_id']
        run_name = model_info['run_name']
        print("Evaluating model: Run ID={run_id}, Run Name={run_name}")

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
            print("Failed to evaluate model with Run ID={run_id}: {e}")
            continue  # Proceed to the next model

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df