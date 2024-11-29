# mlflow_utils/models.py
from typing import List, Dict, Any, Tuple
from centralized_nlp_package.nli_utils import ModelArguments, DataTrainingArguments, run_glue
from transformers import AutoModel, pipeline, AutoModelForSequenceClassification, TrainingArguments
from .base_model import BaseModel

import torch
from loguru import logger
import subprocess

class DeBERTaModel(BaseModel):
    def load_model(self):
        logger.info(f"Loading DeBERTa model from {self.model_path}")
        return pipeline("zero-shot-classification", model=self.model_path, device=self.device)

    def train(self, train_file: str, validation_file: str, param_dict: Dict[str, Any]) -> Tuple[AutoModelForSequenceClassification, Dict[str, float]]:
        logger.info("Starting training for DeBERTa model")

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
            output_dir=param_dict.get("output_dir", "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-base-zeroshot-v2_Santhosh_test/"),
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
        trained_model, eval_metrics = run_glue(model_args, data_args, training_args)

        logger.info("Training completed for DeBERTa model")
        return trained_model, eval_metrics

    def evaluate(self, validation_file: str) -> Dict[str, float]:
        logger.info("Evaluating DeBERTa model")
        # Placeholder for actual evaluation logic
        accuracy = torch.rand(1).item()  # Replace with real evaluation
        logger.info(f"Evaluation accuracy: {accuracy}")
        return {"accuracy": accuracy}

class FinBERTModel(BaseModel):
    def load_model(self):
        logger.info(f"Loading FinBERT model from {self.model_path}")
        return pipeline("zero-shot-classification", model=self.model_path, device=self.device)

    def train(self, train_file: str, validation_file: str, hyperparameters: Dict[str, Any]):
        logger.info("Starting training for FinBERT model")
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
        logger.info("Training completed for FinBERT model")

    def evaluate(self, validation_file: str) -> Dict[str, float]:
        logger.info("Evaluating FinBERT model")
        # Placeholder for actual evaluation logic
        accuracy = torch.rand(1).item()  # Replace with real evaluation
        logger.info(f"Evaluation accuracy: {accuracy}")
        return {"accuracy": accuracy}

def get_model(model_path: str, device: int = 0, **kwargs) -> BaseModel:
    base_model_name = model_path.split('/')[-1]
    model_name = base_model_name.lower()
    if model_name.startswith("deberta"):
        return DeBERTaModel(model_name, model_path, device)
    elif model_name.startswith("finbert"):
        return FinBERTModel(model_name, model_path, device)
    else:
        logger.error(f"Model {model_name} is not supported.")
        raise ValueError(f"Model {model_name} is not supported.")
