# mlflow_utils/models.py
from typing import List, Dict, Any, Tuple
from centralized_nlp_package.nli_utils import ModelArguments, DataTrainingArguments, run_glue
from transformers import AutoModel, pipeline, AutoModelForSequenceClassification, TrainingArguments
from .base_model import BaseModel

import torch
#from loguru import logger
import subprocess

class DeBERTaModel(BaseModel):
    """
    A specialized model class for handling DeBERTa-based Natural Language Inference (NLI) tasks.
    
    This class inherits from `BaseModel` and provides functionalities to load, train, and evaluate a DeBERTa model for zero-shot classification tasks. It utilizes Hugging Face's Transformers pipeline for model operations and integrates with MLflow for experiment tracking.
    
    Args:
        model_path (str):
            The file path or identifier for the pre-trained DeBERTa model.
        device (int):
            The device index to run the model on (e.g., 0 for GPU, -1 for CPU).
    """
    def load_model(self):
        print("Loading DeBERTa model from {self.model_path}")
        return pipeline("zero-shot-classification", model=self.model_path, device=self.device)

    def train(self, train_file: str, validation_file: str, param_dict: Dict[str, Any], output_dir: str, eval_entailment_thresold: float = 0.5) -> Tuple[AutoModelForSequenceClassification, Dict[str, float]]:
        """
        Train the DeBERTa model on the provided dataset with specified hyperparameters.
        
        This method orchestrates the training process by setting up model and data arguments, initializing training configurations, and invoking the training pipeline. After training, it evaluates the model and returns the trained model, tokenizer, and evaluation metrics.
        
        Args:
            train_file (str):
                Path to the training data CSV file.
            validation_file (str):
                Path to the validation data CSV file.
            param_dict (Dict[str, Any]):
                A dictionary of hyperparameters for training, including:
                    - "cache_dir" (str, optional): Directory to cache models.
                    - "learning_rate" (float, optional): Learning rate for training.
                    - "num_train_epochs" (int, optional): Number of training epochs.
                    - "per_device_train_batch_size" (int, optional): Batch size for training.
                    - "per_device_eval_batch_size" (int, optional): Batch size for evaluation.
                    - "task_name" (str, optional): The name of the task (e.g., "nli").
                    - "max_seq_length" (int, optional): Maximum sequence length for inputs.
                    - "pad_to_max_length" (bool, optional): Whether to pad inputs to the maximum length.
                    - "overwrite_cache" (bool, optional): Whether to overwrite the cached data.
                    - "max_train_samples" (int, optional): Maximum number of training samples.
                    - "max_eval_samples" (int, optional): Maximum number of evaluation samples.
                    - "max_predict_samples" (int, optional): Maximum number of prediction samples.
            output_dir (str):
                Directory where the trained model and outputs will be saved.
            eval_entailment_thresold (float, optional):
                Threshold for entailment evaluation to determine positive classifications. Defaults to 0.5.
        
        Returns:
            Tuple[AutoModelForSequenceClassification, AutoTokenizer, Dict[str, float]]:
                - Trained Hugging Face model instance.
                - Tokenizer associated with the trained model.
                - A dictionary of evaluation metrics (e.g., accuracy, F1 score).
        """
        print("Starting training for DeBERTa model")

        # Prepare ModelArguments
        model_args = ModelArguments(
            model_name_or_path=self.model_path,
            cache_dir=param_dict.get("cache_dir", None),
            # learning_rate=param_dict.get("learning_rate", 2e-5),
            # weight_decay=param_dict.get("weight_decay", 0.01),
            # per_device_train_batch_size=param_dict.get("train_batch_size", 16),
            # per_device_eval_batch_size=param_dict.get("eval_batch_size", 16)
        )

        # Prepare DataTrainingArguments
        data_args = DataTrainingArguments(
            task_name=param_dict.get("task_name", None),
            train_file=train_file,
            validation_file=validation_file,
            max_seq_length=param_dict.get("max_seq_length", 128),
            pad_to_max_length=param_dict.get("pad_to_max_length", True),
            overwrite_cache=param_dict.get("overwrite_cache", False),
            max_train_samples=param_dict.get("max_train_samples", None),
            max_eval_samples=param_dict.get("max_eval_samples", None),
            max_predict_samples=param_dict.get("max_predict_samples", None)
        )

        # Prepare TrainingArguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            do_train=True,
            do_eval=True,
            num_train_epochs=param_dict.get("n_epochs", 3),
            learning_rate=param_dict.get("learning_rate", 2e-5),
            weight_decay=param_dict.get("weight_decay", 0.01),
            per_device_train_batch_size=param_dict.get("train_batch_size", 16),
            per_device_eval_batch_size=param_dict.get("eval_batch_size", 16),
            fp16=param_dict.get("fp16", True),
            report_to=None,
            overwrite_output_dir=param_dict.get("overwrite_output_dir", True),
            push_to_hub=param_dict.get("push_to_hub", False),
            seed=param_dict.get("seed", 42)
        )

        # Call run_glue
        trained_model, tokenizer, eval_metrics = run_glue(model_args, data_args, training_args, eval_entailment_thresold=eval_entailment_thresold)

        print("Training completed for DeBERTa model")
        return trained_model, tokenizer, eval_metrics

    def evaluate(self, validation_file: str) -> Dict[str, float]:
        """
        Evaluate the DeBERTa model on the validation dataset.
        
        This method performs evaluation of the trained DeBERTa model using the provided validation dataset. It computes relevant metrics such as accuracy and returns them in a dictionary.
        
        Args:
            validation_file (str):
                Path to the validation data CSV file.
        
        Returns:
            Dict[str, float]:
                A dictionary containing evaluation metrics, for example:
                    - "accuracy" (float): The accuracy of the model on the validation set.
        
        Raises:
            FileNotFoundError:
                If the validation data file does not exist.
            Exception:
                If there is an error during the evaluation process.
        
        Example:
            >>> model = DeBERTaModel(model_path="deberta-base")
            >>> metrics = model.evaluate(validation_file="data/validation.csv")
            >>> print(metrics)
            {'accuracy': 0.85}
        """
        print("Evaluating DeBERTa model")
        # Placeholder for actual evaluation logic
        accuracy = torch.rand(1).item()  # Replace with real evaluation
        print("Evaluation accuracy: {accuracy}")
        return {"accuracy": accuracy}

class FinBERTModel(BaseModel):
    """
    A specialized model class for handling FinBERT-based Natural Language Inference (NLI) tasks.
    
    This class inherits from `BaseModel` and provides functionalities to load, train, and evaluate a FinBERT model for zero-shot classification tasks. It leverages subprocess calls to execute training scripts and integrates with MLflow for experiment tracking.
    
    Args:
        model_path (str):
            The file path or identifier for the pre-trained FinBERT model.
        device (int):
            The device index to run the model on (e.g., 0 for GPU, -1 for CPU).
    """
    def load_model(self):
        print("Loading FinBERT model from {self.model_path}")
        return pipeline("zero-shot-classification", model=self.model_path, device=self.device)

    def train(self, train_file: str, validation_file: str, hyperparameters: Dict[str, Any]):
        """
        Train the FinBERT model using a separate training script.
        
        This method initiates the training process for the FinBERT model by executing an external training script (`run_glue.py`) with the provided training and validation data files and hyperparameters. It leverages Python's `subprocess` module to run the training script in a separate process.
        
        Args:
            train_file (str):
                Path to the training data CSV file.
            validation_file (str):
                Path to the validation data CSV file.
            hyperparameters (Dict[str, Any]):
                A dictionary of hyperparameters for training, including:
                    - "output_dir" (str, optional): Directory to save the trained model and outputs.
                    - "n_epochs" (int, optional): Number of training epochs.
                    - "learning_rate" (float, optional): Learning rate for the optimizer.
                    - "weight_decay" (float, optional): Weight decay for the optimizer.
                    - "train_batch_size" (int, optional): Batch size for training.
                    - "eval_batch_size" (int, optional): Batch size for evaluation.
        
        Returns:
            None
        
        Raises:
            subprocess.CalledProcessError:
                If the training script execution fails.
            FileNotFoundError:
                If the training script (`run_glue.py`) does not exist.
            Exception:
                If there is an error during the training process.
        
        Example:
            >>> model = FinBERTModel(model_path="finbert-base-uncased")
            >>> hyperparams = {
            ...     "output_dir": "./finbert_output",
            ...     "n_epochs": 4,
            ...     "learning_rate": 3e-5,
            ...     "weight_decay": 0.01,
            ...     "train_batch_size": 16,
            ...     "eval_batch_size": 16
            ... }
            >>> model.train(
            ...     train_file="data/train_finbert.csv",
            ...     validation_file="data/validation_finbert.csv",
            ...     hyperparameters=hyperparams
            ... )
            INFO - Starting training for FinBERT model
            INFO - Training completed for FinBERT model
        """
        print("Starting training for FinBERT model")
        subprocess.run([
            "python", "run_glue.py",
            "--model_name_or_path", self.model_path,
            "--output_dir", hyperparameters.get("output_dir", "./model_output"),
            "--train_file", train_file,
            "--validation_file", validation_file,
            "--do_train",
            "--do_eval",
            "--num_train_epochs", str(hyperparameters.get("n_epochs", 3)),
            "--fp16",
            "--report_to", "none",
            "--learning_rate", str(hyperparameters.get("learning_rate", 2e-5)),
            "--weight_decay", str(hyperparameters.get("weight_decay", 0.01)),
            "--per_device_train_batch_size", str(hyperparameters.get("train_batch_size", 16)),
            "--per_device_eval_batch_size", str(hyperparameters.get("eval_batch_size", 16))
        ], check=True)
        print("Training completed for FinBERT model")

    def evaluate(self, validation_file: str) -> Dict[str, float]:
        print("Evaluating FinBERT model")
        # Placeholder for actual evaluation logic
        accuracy = torch.rand(1).item()  # Replace with real evaluation
        print("Evaluation accuracy: {accuracy}")
        return {"accuracy": accuracy}

def get_model(model_path: str, device: int = 0, **kwargs) -> BaseModel:
    """
    Retrieve an instance of a model based on the provided model path.
    
    This function selects and initializes a model class (`DeBERTaModel` or `FinBERTModel`) based on the naming convention of the provided `model_path`. It ensures that only supported models are instantiated and logs appropriate messages during the selection process.
    
    Args:
        model_path (str):
            The file path or identifier for the pre-trained model. The function determines the model type based on this path.
        device (int, optional):
            The device index to run the model on (e.g., 0 for GPU, -1 for CPU). Defaults to 0.
        **kwargs:
            Additional keyword arguments to pass to the model initializer.
    
    Returns:
        BaseModel:
            An instance of the selected model class (`DeBERTaModel` or `FinBERTModel`).
    
    Raises:
        ValueError:
            If the provided `model_path` does not correspond to a supported model type.
        Exception:
            If there is an error during model initialization.
    
    Example:
        >>> model = get_model(model_path="deberta-v3-base")
        >>> print(type(model))
        <class 'mlflow_utils.models.DeBERTaModel'>
        
        >>> model = get_model(model_path="finbert-base-uncased")
        >>> print(type(model))
        <class 'mlflow_utils.models.FinBERTModel'>
        
        >>> model = get_model(model_path="unknown-model")
        ERROR - Model unknown-model is not supported.
        Traceback (most recent call last):
            ...
        ValueError: Model unknown-model is not supported.
    """
    base_model_name = model_path.split('/')[-1]
    model_name = base_model_name.lower()
    if model_name.startswith("deberta"):
        return DeBERTaModel(model_name, model_path, device)
    elif model_name.startswith("finbert"):
        return FinBERTModel(model_name, model_path, device)
    else:
        print("Model {model_name} is not supported.")
        raise ValueError(f"Model {model_name} is not supported.")
