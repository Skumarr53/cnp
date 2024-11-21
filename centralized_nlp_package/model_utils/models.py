# mlflow_utils/models.py

from .base_model import BaseModel
from transformers import AutoModel, pipeline
import torch
from loguru import logger
import subprocess

class DeBERTaModel(BaseModel):
    def load_model(self):
        logger.info(f"Loading DeBERTa model from {self.model_path}")
        return pipeline("zero-shot-classification", model=self.model_path, device=self.device)

    def train(self, train_file: str, validation_file: str, hyperparameters: Dict[str, Any]):
        logger.info("Starting training for DeBERTa model")
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
        logger.info("Training completed for DeBERTa model")

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

def get_model(model_name: str, model_path: str, device: int = 0, **kwargs) -> BaseModel:
    model_name = model_name.lower()
    if model_name == "deberta":
        return DeBERTaModel(model_name, model_path, device)
    elif model_name == "finbert":
        return FinBERTModel(model_name, model_path, device)
    else:
        logger.error(f"Model {model_name} is not supported.")
        raise ValueError(f"Model {model_name} is not supported.")
