# mlflow_utils/base_model.py

from abc import ABC, abstractmethod
from typing import Any, Dict
import torch
from transformers import pipeline
import mlflow.pytorch
from loguru import logger

class BaseModel(ABC):
    def __init__(self, model_name: str, model_path: str, device: int = 0):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else -1
        self.model = self.load_model()

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def train(self, train_file: str, validation_file: str, hyperparameters: Dict[str, Any]):
        pass

    @abstractmethod
    def evaluate(self, validation_file: str) -> Dict[str, float]:
        pass

    def save_model(self, output_dir: str):
        logger.info(f"Saving model to {output_dir}")
        mlflow.pytorch.log_model(self.model, "model")
