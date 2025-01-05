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
from .model_selector import list_available_models
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
        self.pred_path = "predictions.csv"
        self.testset_name, _ = os.path.splitext(os.path.basename(self.validation_file))
         
        
        mlflow.set_experiment(self.experiment_name)

        self.runs_list = self.get_run_names()
        logger.info(f"Experiment set to {self.experiment_name}")

    def run_single_experiment(self, run_name, base_model, base_model_name, dataset_version, dataset_name, param_set):
        torch.cuda.empty_cache()

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

            nli_pipeline = pipeline("zero-shot-classification", model=ft_model, 
                                    tokenizer=tokenizer, batch_size=2,
                                    device= 0 if torch.cuda.is_available() else -1)

            eval_df = self.eval_df.copy()
            metrics = get_nli_model_metrics(nli_pipeline, eval_df, self.eval_entailment_thresold)
                    
            # Example metrics dictionary

            eval_df['entailment_scores'] = metrics['scores']  
            eval_df['predictions'] = metrics['predictions']
            eval_df.to_csv(self.pred_path, index=False)


            components = {
                "model": ft_model,
                "tokenizer": tokenizer
                }

            # Log multiple metrics at once
            mlflow.log_metrics({
                "accuracy": metrics['accuracy'],
                "f1_score": metrics['f1_score'],
                "precision": metrics['precision'],
                "recall": metrics['recall'],
                "roc_auc": metrics['roc_auc']
            })
            mlflow.log_artifact(self.pred_path)
            mlflow.transformers.log_model(
                transformers_model=components,
                task="zero-shot-classification",
                artifact_path="model")
            logger.info(f"Model Artifacts logged successfully")
            
            logger.info(f"Run {run_name} completed with accuracy: {eval_metrics['eval_accuracy']}")
        except Exception as e:
            logger.error(f"Failed during run {run_name}: {e}")

        finally:
            # Cleanup to free memory
            del components
            del nli_pipeline
            del ft_model
            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()

    def run_experiments(self):
        
        for base_model in self.base_model_versions:

            base_model_name = base_model.split('/')[-1]

            if self.evalute_pretrained_model:
                self.evaluate_pretrained_model(base_model)

            for dataset_version in self.dataset_versions:
                for idx, param_set in enumerate(self.hyperparameters):
                    dataset_name = dataset_version.split('.')[0]

                    run_name = f"{base_model_name}_{dataset_name}_{self.testset_name}_param_set{idx+1}"
                    if run_name in self.runs_list:
                        logger.info(f"Skipping {run_name} as it already exists")
                        continue
                    with mlflow.start_run(run_name=run_name) as run:
                        self.run_single_experiment(
                            run_name, 
                            base_model, 
                            base_model_name, 
                            dataset_version, 
                            dataset_name, 
                            param_set
                        )




    def evaluate_pretrained_model(self, base_model):

        base_model_name = base_model.split('/')[-1]
        
        pretrained_run_name = f"{base_model_name}_{self.testset_name}_pretrained"
        if pretrained_run_name not in self.runs_list: 
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

                nli_pipeline = pipeline("zero-shot-classification", 
                                        model=model, tokenizer=tokenizer,
                                        batch_size=2,
                                        device= 0 if torch.cuda.is_available() else -1)

                eval_df = self.eval_df.copy()
                metrics = get_nli_model_metrics(nli_pipeline, eval_df, self.eval_entailment_thresold)
                
                print("metrics",metrics)
                
                eval_df['entailment_scores'] = metrics['scores']  
                eval_df['predictions'] = metrics['predictions']
                eval_df.to_csv(self.pred_path, index=False)

                components = {
                    "model": model,
                    "tokenizer": tokenizer
                    }

                mlflow.transformers.log_model(
                                transformers_model=components,
                                task="zero-shot-classification",
                                artifact_path="model")

                
                # Log metrics 
                mlflow.log_metrics({
                                    "accuracy": metrics['accuracy'],
                                    "f1_score": metrics['f1_score'],
                                    "precision": metrics['precision'],
                                    "recall": metrics['recall'],
                                    "roc_auc": metrics['roc_auc']
                                })
                mlflow.log_artifact(self.pred_path)

                logger.info(f"Run {pretrained_run_name} completed with metrics: {metrics}")
                
                del components
                del nli_pipeline
                del model
                del tokenizer
                torch.cuda.empty_cache()
                gc.collect()
    
    def get_run_names(self):
        runs_list = list_available_models(self.experiment_name)
        return [run['run_name'] for run in runs_list]


            
