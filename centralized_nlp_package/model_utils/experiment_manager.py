# mlflow_utils/experiment_manager.py
import os
import torch
import pandas as pd
from loguru import logger
from centralized_nlp_package.common_utils import get_current_date_str
from centralized_nlp_package.nli_utils import get_nli_model_metrics
from datetime import datetime
import mlflow
# from .utils import format_experiment_name, format_run_name, get_current_date, validate_path
from typing import List, Dict, Any
from transformers import pipeline
from .models import get_model


class ExperimentManager:
    def __init__(
        self,
        base_name: str,
        data_src: str,
        dataset_versions: List[str],
        hyperparameters: List[Dict[str, Any]],
        base_model_versions: str,
        output_dir: str,
        train_file: str,
        validation_file: str,
        user_id: str = 'santhosh.kumar3@voya.com',
        evalute_pretrained_model: bool = True,
        **kwargs
    ):

        self.run_date = get_current_date_str() #datetime.today().strftime('%Y%m%d') # get_current_date()
        self.experiment_name = f"/Users/{user_id}/{base_name}_{data_src}_{self.run_date}"
        self.dataset_versions = dataset_versions
        self.hyperparameters = hyperparameters
        self.base_model_versions = base_model_versions
        self.output_dir =output_dir 
        self.validation_file = validation_file
        self.train_file = train_file
        self.evalute_pretrained_model = evalute_pretrained_model
         
        
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"Experiment set to {self.experiment_name}")

    def run_experiments(self):
        for base_model in self.base_model_versions:

            base_model_name = base_model.split('/')[-1]

            if self.evalute_pretrained_model:
                self.evaluate_pretrained_model(base_model, param_set)

            for dataset_version in self.dataset_versions:
                for idx, param_set in enumerate(self.hyperparameters):
                    run_name = f"{base_model_name}_{dataset_version}_param_set{idx+1}"
                    with mlflow.start_run(run_name=run_name) as run:
                        logger.info(f"Starting finetuning run: {run_name}")
                        mlflow.set_tag("run_date", self.run_date)
                        mlflow.set_tag("base_model_name", base_model_name)
                        mlflow.set_tag("dataset_version", dataset_version)
                        mlflow.set_tag("run_type", "finetuned")

                        # Log hyperparameters
                        mlflow.log_params({
                            "num_train_epochs": param_set.get("n_epochs", 3),
                            "learning_rate": param_set.get("learning_rate", 2e-5),
                            "weight_decay": param_set.get("weight_decay", 0.01),
                            "per_device_train_batch_size": param_set.get("train_batch_size", 16)
                        })
                        
                        # Initialize model
                        model = get_model(
                            model_path=base_model,
                            device=0 if torch.cuda.is_available() else -1
                        )
                        
                        # Train model
                        train_file_path = self.train_file.format(data_version=dataset_version)
                        trained_model, eval_metrics = model.train(
                            train_file=train_file_path,
                            validation_file=self.validation_file,
                            param_dict=param_set
                        )
                        
                        # Save and log model
                        # output_path = os.path.join(self.output_dir, run_name)
                        # os.makedirs(output_path, exist_ok=True)
                        # trained_model.save_model(output_path)

                        mlflow.pytorch.log_model(trained_model, "model")
                    
                        # for metric_name, metric_value in metrics.items():
                        mlflow.log_metric('accuracy', eval_metrics['eval_accuracy'])
                        
                        logger.info(f"Run {run_name} completed with metrics: {eval_metrics['eval_accuracy']}")

    def evaluate_pretrained_model(self, base_model, param_set):

        base_model_name = base_model.split('/')[-1]

        eval_df = pd.read_csv(self.validation_file)
        
        pretrained_run_name = f"{base_model_name}_pretrained"
        with mlflow.start_run(run_name=pretrained_run_name) as pretrained_run:
        
            logger.info(f"Starting pretrained evaluation run: {pretrained_run_name}")
            mlflow.set_tag("run_date", self.run_date)
            mlflow.set_tag("base_model_name", base_model_name)
            mlflow.set_tag("dataset_version", 'NA')
            mlflow.set_tag("run_type", "pretrained")


            # Log parameters (same as finetuned for consistency)
            mlflow.log_params({
                "num_train_epochs": 0,  # No training
                "learning_rate": 0.0,
                "weight_decay": 0.0,
                "per_device_train_batch_size": param_set.get("train_batch_size", 16)
            })

            nli_pipeline = pipeline("zero-shot-classification", model = base_model, device= 0 if torch.cuda.is_available() else -1)

            metrics = get_nli_model_metrics(nli_pipeline, eval_df)
            
            mlflow.pytorch.log_model(nli_pipeline.model, "model")
            
            # for metric_name, metric_value in metrics.items():
            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            logger.info(f"Run {pretrained_run_name} completed with metrics: {metrics}")
            
