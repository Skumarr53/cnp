# mlflow_utils/experiment_manager.py

import mlflow
from .utils import format_experiment_name, format_run_name, get_current_date, validate_path
from .models import get_model
from loguru import logger
from typing import List, Dict, Any
import os

class ExperimentManager:
    def __init__(
        self,
        base_name: str,
        data_src: str,
        dataset_versions: List[str],
        hyperparameters: List[Dict[str, Any]],
        base_model: str,
        model_path: str,
        output_dir: str,
        train_file: str,
        validation_file: str,
        **kwargs
    ):
        self.base_name = base_name
        self.data_src = data_src
        self.dataset_versions = dataset_versions
        self.hyperparameters = hyperparameters
        self.base_model = base_model
        self.model_path = validate_path(model_path)
        self.output_dir = validate_path(output_dir)
        self.train_file = validate_path(train_file)
        self.validation_file = validate_path(validation_file)
        self.run_date = get_current_date()
        self.experiment_name = format_experiment_name(self.base_name, self.data_src, self.run_date)
        self.run_name_base = self.base_name
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"Experiment set to {self.experiment_name}")

    def run_experiments(self):
        for dataset in self.dataset_versions:
            for idx, param_set in enumerate(self.hyperparameters):
                run_name = format_run_name(
                    self.run_name_base, self.run_date, dataset, idx
                )
                with mlflow.start_run(run_name=run_name) as run:
                    logger.info(f"Starting run: {run_name}")
                    mlflow.set_tag("run_date", self.run_date)
                    # Log parameters
                    mlflow.log_param("run_date", self.run_date)
                    mlflow.log_param("dataset", dataset)
                    mlflow.log_param("base_model_name", self.base_model)
                    mlflow.log_param("num_train_epochs", param_set.get("n_epochs", 3))
                    mlflow.log_param("learning_rate", param_set.get("learning_rate", 2e-5))
                    mlflow.log_param("weight_decay", param_set.get("weight_decay", 0.01))
                    mlflow.log_param("per_device_train_batch_size", param_set.get("train_batch_size", 16))
                    
                    # Initialize model
                    model = get_model(
                        model_name=self.base_model,
                        model_path=self.model_path,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    
                    # Train model
                    model.train(
                        train_file=self.train_file,
                        validation_file=self.validation_file,
                        hyperparameters=param_set
                    )
                    
                    # Save and log model
                    model.save_model(self.output_dir)
                    
                    # Evaluate model
                    metrics = model.evaluate(self.validation_file)
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)
                    
                    logger.info(f"Run {run_name} completed with metrics: {metrics}")
