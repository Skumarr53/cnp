# mlflow_utils/experiment_manager.py
import os
import gc
import torch
import pandas as pd
from loguru import logger
from centralized_nlp_package.common_utils import get_current_date_str
from centralized_nlp_package.nli_utils import get_nli_model_metrics
from datetime import datetime
import mlflow
import mlflow.transformers
from accelerate import Accelerator
# from .utils import format_experiment_name, format_run_name, get_current_date, validate_path
from typing import List, Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from .models import get_model


# mlflow.set_tracking_uri("http://localhost:5000")

class ExperimentManager:
    def __init__(
        self,
        base_name: str,
        data_src: str,
        dataset_versions: List[str],
        hyperparameters: List[Dict[str, Any]],
        base_model_versions: str,
        train_file: str,
        validation_file: str,
        evalute_pretrained_model: bool = True,
        eval_entailment_thresold: float = 0.5,
        user_id: str = 'santhosh.kumar3@voya.com',
        output_dir: str = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-base-zeroshot-v2_Santhosh_test/",
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
        self.eval_entailment_thresold = eval_entailment_thresold
        self.eval_df = pd.read_csv(validation_file)
        self.accelerator = Accelerator()
         
        
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"Experiment set to {self.experiment_name}")

    def run_experiments(self):
        pred_path = "predictions.csv"
        for base_model in self.base_model_versions:

            base_model_name = base_model.split('/')[-1]

            if self.evalute_pretrained_model:
                self.evaluate_pretrained_model(base_model)

            for dataset_version in self.dataset_versions:
                for idx, param_set in enumerate(self.hyperparameters):
                    dataset_name = dataset_version.split('.')[0]

                    run_name = f"{base_model_name}_{dataset_name}_param_set{idx+1}"
                    with mlflow.start_run(run_name=run_name) as run:
                        try:
                            logger.info(f"Starting finetuning run: {run_name}")
                            mlflow.set_tag("run_date", self.run_date)
                            mlflow.set_tag("base_model_name", base_model_name)
                            mlflow.set_tag("dataset_version", dataset_name)
                            mlflow.set_tag("run_type", "finetuned")

                            # Log hyperparameters
                            mlflow.log_params({
                                "eval_entailment_thresold": self.eval_entailment_thresold,
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
                            ft_model, tokenizer, eval_metrics = model.train(
                                train_file=train_file_path,
                                validation_file=self.validation_file,
                                param_dict=param_set,
                                output_dir = self.output_dir,
                                eval_entailment_thresold=self.eval_entailment_thresold
                            )


                            ## Prepare logging metrics
                            # Example metrics dictionary
                            metrics = {
                                "accuracy": eval_metrics['eval_accuracy'],
                                "f1_score": eval_metrics['eval_f1_score'],
                                "precision": eval_metrics['eval_precision'],
                                "recall": eval_metrics['eval_recall'],
                                "roc_auc": eval_metrics['eval_roc_auc']
                            }

                            # Predictions
                            eval_df = self.eval_df.copy()
                            eval_df['entailment_scores'] = metrics['eval_scores']          
                            eval_df['predictions'] = metrics['eval_predictions']
                            eval_df.to_csv(pred_path, index=False)

                            components = {
                                "model": ft_model,
                                "tokenizer": tokenizer
                                }

                            # Log multiple metrics at once
                            mlflow.log_metrics(metrics)
                            mlflow.log_artifact(pred_path)
                            mlflow.transformers.log_model(
                                transformers_model=components,
                                task="zero-shot-classification",
                                artifact_path="model")
                            logger.info(f"Model Artifacts logged successfully")
                            
                            logger.info(f"Run {run_name} completed with accuracy: {eval_metrics['eval_accuracy']}")
                            mlflow.end_run(status="SUCCESS")
                        except Exception as e:
                            logger.error(f"Error occurred during run {run_name}: {str(e)}")
                            mlflow.end_run(status="FAILED")


    def evaluate_pretrained_model(self, base_model):

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
                "eval_entailment_thresold": self.eval_entailment_thresold,
                "num_train_epochs": 0,  # No training
                "learning_rate": 0.0,
                "weight_decay": 0.0,
                "per_device_train_batch_size": 16
            })

            ## load model
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForSequenceClassification.from_pretrained(base_model)

            nli_pipeline = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer, device= 0 if torch.cuda.is_available() else -1)

            metrics = get_nli_model_metrics(nli_pipeline, eval_df, self.eval_entailment_thresold)
            
            print("metrics",metrics)

            components = {
                "model": model,
                "tokenizer": tokenizer
                }

            mlflow.transformers.log_model(
                            transformers_model=components,
                            task="zero-shot-classification",
                            artifact_path="model")

            
            # Log metrics 
            mlflow.log_metrics(metrics)

            logger.info(f"Run {pretrained_run_name} completed with metrics: {metrics}")
            
