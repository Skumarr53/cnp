# mlflow_utils/experiment_manager.py
import os
import gc
import torch
import pandas as pd
#from loguru import logger
from sklearn.model_selection import KFold


from datetime import datetime
import mlflow
import mlflow.transformers
from accelerate import Accelerator
# from .utils import format_experiment_name, format_run_name, get_current_date, validate_path
from typing import List, Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from .model_selector import list_available_models
from .models import get_model
from .model_evaluation import  plot_conf_matrix, generate_and_plot_confusion_matrices
from centralized_nlp_package.common_utils import get_current_date_str
from centralized_nlp_package.nli_utils import get_nli_model_metrics


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
        """
        Initialize the ExperimentManager with the specified configuration.

        This constructor sets up the ExperimentManager by initializing experiment parameters, loading the validation dataset, and configuring MLflow for experiment tracking. It also sets up necessary components such as the Accelerator for handling hardware acceleration.

        Args:
            base_name (str): 
                The base name for the experiment.
            data_src (str): 
                The source of the data being used.
            dataset_versions (List[str]): 
                A list of dataset version identifiers to be used in the experiments.
            hyperparameters (List[Dict[str, Any]]): 
                A list of dictionaries, each containing a set of hyperparameters for a specific experiment run.
            base_model_versions (str): 
                The versions of the base models to be fine-tuned.
            train_file (str): 
                Path to the training data file. It should be a string that can be formatted with a dataset version.
            validation_file (str): 
                Path to the validation data file.
            evalute_pretrained_model (bool, optional): 
                Flag indicating whether to evaluate the pretrained model before fine-tuning. Defaults to True.
            eval_entailment_thresold (float, optional): 
                The threshold for entailment evaluation. Defaults to 0.5.
            user_id (str, optional): 
                The user identifier, typically an email address. Defaults to 'santhosh.kumar3@voya.com'.
            output_dir (str, optional): 
                Directory path where the output (e.g., trained models) will be saved. Defaults to the specified DBFS path.
            **kwargs: 
                Additional keyword arguments that can be used to customize the ExperimentManager.

        Attributes:
            run_date (str): 
                The date when the experiment run was initiated, in 'YYYYMMDD' format.
            experiment_name (str): 
                The name of the MLflow experiment, constructed from user ID, base name, data source, and run date.
            dataset_versions (List[str]): 
                List of dataset versions to be used.
            hyperparameters (List[Dict[str, Any]]): 
                List of hyperparameter sets for experiments.
            base_model_versions (str): 
                Versions of base models to fine-tune.
            output_dir (str): 
                Directory path for saving outputs.
            validation_file (str): 
                Path to the validation data file.
            train_file (str): 
                Path to the training data file.
            evalute_pretrained_model (bool): 
                Flag to evaluate pretrained models.
            eval_entailment_thresold (float): 
                Entailment threshold for evaluation.
            eval_df (pd.DataFrame): 
                The validation dataset loaded into a Pandas DataFrame.
            accelerator (Accelerator): 
                The Accelerator instance for handling hardware acceleration.
            pred_path (str): 
                Path to save prediction results.
            testset_name (str): 
                Name of the test dataset, derived from the validation file name.
            runs_list (List[str]): 
                List of existing run names in the MLflow experiment.

        Example:
            >>> from mlflow_utils.experiment_manager import ExperimentManager
            >>> hyperparams = [
            ...     {"n_epochs": 3, "learning_rate": 2e-5, "weight_decay": 0.01, "train_batch_size": 16},
            ...     {"n_epochs": 5, "learning_rate": 3e-5, "weight_decay": 0.02, "train_batch_size": 32}
            ... ]
            >>> dataset_versions = ["v1.0", "v1.1"]
            >>> manager = ExperimentManager(
            ...     base_name="NLI_Finetune",
            ...     data_src="source_A",
            ...     dataset_versions=dataset_versions,
            ...     hyperparameters=hyperparams,
            ...     base_model_versions="bert-base-uncased",
            ...     train_file="path/to/train.csv",
            ...     validation_file="path/to/validation.csv"
            ... )
        """

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
        print("Experiment set to {self.experiment_name}")

    def run_single_experiment(self, run_name, base_model, base_model_name, dataset_version, dataset_name, param_set):
        """
        Execute a single fine-tuning experiment run.

        This method performs the fine-tuning of a base model on a specified dataset using provided hyperparameters. It handles the training process, evaluates the fine-tuned model, logs metrics and artifacts to MLflow, and manages resource cleanup.

        Args:
            run_name (str): 
                The name of the experiment run.
            base_model (str): 
                The path or identifier of the base model to be fine-tuned.
            base_model_name (str): 
                The name of the base model, extracted from its path or identifier.
            dataset_version (str): 
                The version identifier of the dataset to be used for training.
            dataset_name (str): 
                The name of the dataset, derived from the dataset version.
            param_set (Dict[str, Any]): 
                A dictionary containing the hyperparameters for the fine-tuning run, such as number of epochs, learning rate, weight decay, and batch size.

        Returns:
            None

        Raises:
            Exception: 
                If any step during the experiment run (e.g., training, evaluation, logging) fails.

        Example:
            >>> run_name = "bert-base-v1.0_param_set1"
            >>> base_model = "bert-base-uncased"
            >>> base_model_name = "bert-base-uncased"
            >>> dataset_version = "v1.0"
            >>> dataset_name = "dataset_v1"
            >>> param_set = {"n_epochs": 3, "learning_rate": 2e-5, "weight_decay": 0.01, "train_batch_size": 16}
            >>> manager.run_single_experiment(run_name, base_model, base_model_name, dataset_version, dataset_name, param_set)
        """
        torch.cuda.empty_cache()

        try:
            print("Starting finetuning run: {run_name}")
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

            # Log the Confusion matirx plot as an artifact in MLflow
            plot_names = generate_and_plot_confusion_matrices(eval_df, 'label_GT', 'predictions','sentence2', plot_conf_matrix)
            for filename in plot_names:
                mlflow.log_artifact(filename)
                os.remove(filename)
            mlflow.transformers.log_model(
                transformers_model=components,
                task="zero-shot-classification",
                artifact_path="model")
            print("Model Artifacts logged successfully")
            
            print("Run {run_name} completed with accuracy: {eval_metrics['eval_accuracy']}")
        except Exception as e:
            print("Failed during run {run_name}: {e}")

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
        """
        Execute all configured experiments based on the provided dataset versions and hyperparameters.

        This method orchestrates the entire experiment workflow by iterating over the specified base models, dataset versions, and hyperparameter sets. For each combination, it checks if the experiment run already exists, evaluates pretrained models if required, and initiates new fine-tuning runs while logging results to MLflow.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: 
                If any experiment run fails during execution.

        Example:
            >>> manager.run_experiments()
        """
        
        for base_model in self.base_model_versions:

            base_model_name = base_model.split('/')[-1]

            if self.evalute_pretrained_model:
                self.evaluate_pretrained_model(base_model)

            for dataset_version in self.dataset_versions:
                for idx, param_set in enumerate(self.hyperparameters):
                    dataset_name = dataset_version.split('.')[0]

                    run_name = f"{base_model_name}_{dataset_name}_{self.testset_name}_param_set{idx+1}"
                    if run_name in self.runs_list:
                        print("Skipping {run_name} as it already exists")
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
        """
        Evaluate the performance of a pretrained model without fine-tuning.

        This method assesses the pretrained base model on the validation dataset using the specified entailment threshold. It logs the evaluation metrics and artifacts to MLflow, enabling comparison between pretrained and fine-tuned models.

        Args:
            base_model (str): 
                The path or identifier of the pretrained model to be evaluated.

        Returns:
            None

        Raises:
            Exception: 
                If the evaluation process fails, such as issues with model loading or metric computation.

        Example:
            >>> base_model = "bert-base-uncased"
            >>> manager.evaluate_pretrained_model(base_model)
        """

        base_model_name = base_model.split('/')[-1]
        
        pretrained_run_name = f"{base_model_name}_{self.testset_name}_pretrained"
        if pretrained_run_name not in self.runs_list: 
            with mlflow.start_run(run_name=pretrained_run_name) as pretrained_run:
            
                print("Starting pretrained evaluation run: {pretrained_run_name}")
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

                print("Run {pretrained_run_name} completed with metrics: {metrics}")
                
                del components
                del nli_pipeline
                del model
                del tokenizer
                torch.cuda.empty_cache()
                gc.collect()
    
    def get_run_names(self):
        """
        Retrieve a list of existing run names from the configured MLflow experiment.

        This method queries MLflow to obtain the names of all runs associated with the current experiment. It is used to prevent duplicate runs by checking if a run with the same name already exists.

        Args:
            None

        Returns:
            List[str]: 
                A list of run names present in the MLflow experiment.

        Raises:
            Exception: 
                If there is an issue accessing MLflow runs.

        Example:
            >>> existing_runs = manager.get_run_names()
            >>> print(existing_runs)
            ['bert-base-v1.0_param_set1', 'bert-base-v1.0_param_set2']
        """
        runs_list = list_available_models(self.experiment_name)
        return [run['run_name'] for run in runs_list]



def perform_kfold_training(data_path, base_exp_name, data_src, model_version, hyperparameters, user_id, n_splits=5, random_state=42):
    """
    Perform K-Fold cross-validation training for Natural Language Inference (NLI) tasks using the specified model and hyperparameters.

    This function executes K-Fold cross-validation by splitting the dataset into training and testing subsets for each fold. For each split, it initializes an `ExperimentManager` instance to manage the experiment run, conducts training, evaluates the model, and logs the results using MLflow.

    Args:
        data_path (str):
            Path to the CSV data file containing the dataset.
        base_exp_name (str):
            Base name for the experiment, used to construct the MLflow experiment name.
        data_src (str):
            Identifier for the data source.
        model_version (str):
            Version identifier of the base model to be fine-tuned.
        hyperparameters (Dict[str, Any]):
            Dictionary containing hyperparameters for training, such as learning rate, number of epochs, weight decay, and batch size.
        user_id (str):
            User identifier, typically an email address, used in constructing the experiment name.
        n_splits (int, optional):
            Number of folds for K-Fold cross-validation. Defaults to 5.
        random_state (int, optional):
            Seed for random number generator to ensure reproducibility. Defaults to 42.

    Returns:
        None

    Raises:
        FileNotFoundError:
            If the data file specified by `data_path` does not exist.
        pd.errors.EmptyDataError:
            If the data file is empty.
        Exception:
            If any error occurs during data loading, splitting, or experiment execution.

    Example:
        >>> perform_kfold_training(
        ...     data_path="data/nli_dataset.csv",
        ...     base_exp_name="NLI_Experiment",
        ...     data_src="source_A",
        ...     model_version="bert-base-uncased",
        ...     hyperparameters={
        ...         "n_epochs": 3,
        ...         "learning_rate": 2e-5,
        ...         "weight_decay": 0.01,
        ...         "train_batch_size": 16
        ...     },
        ...     user_id="user@example.com",
        ...     n_splits=5,
        ...     random_state=42
        ... )
        INFO - Initialized KFold with 5 splits.
        INFO - Fold 0: Training and test data prepared.
        INFO - Experiment set to /Users/user@example.com/NLI_Experiment_source_A_20250127
        INFO - Starting finetuning run: bert-base-uncased_dataset_v1_param_set1
        ...
        INFO - Fold 0: Experiment completed.
        ...
    """
    # Load data
    data = pd.read_csv(data_path)
    grouped_data = data.groupby('sentence1')
    pairs = [group for _, group in grouped_data]

    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    print("Initialized KFold with {n_splits} splits.")


    # Get the directory of the input data
    input_dir = os.path.dirname(data_path)

    # Iterate over each fold
    for fold, (train_index, test_index) in enumerate(kf.split(pairs)):
        train_pairs = [pairs[i] for i in train_index]
        test_pairs = [pairs[i] for i in test_index]
        print(f"Processing fold {fold}")

        # Concatenate the pairs back into DataFrames
        train_data = pd.concat(train_pairs).reset_index(drop=True)
        test_data = pd.concat(test_pairs).reset_index(drop=True)
        print("Fold {fold}: Training and test data prepared.")


        # Save the split data to temporary files in the same directory as the input data
        train_fold = f"train_fold_{fold}.csv"
        train_file_path = os.path.join(input_dir, train_fold)
        test_file_path = os.path.join(input_dir, f"test_fold_{fold}.csv")
        train_data.to_csv(train_file_path, index=False)
        test_data.to_csv(test_file_path, index=False)


        # Initialize the ExperimentManager for this fold
        experiment_manager = ExperimentManager(
            base_name=base_exp_name,
            data_src=data_src,
            dataset_versions=[train_fold],
            hyperparameters=[hyperparameters],
            base_model_versions=[model_version],
            train_file=train_file_path,
            validation_file=test_file_path,
            evalute_pretrained_model=False,
            eval_entailment_thresold=0.8,
            user_id=user_id
        )


        # Run the experiment for this fold
        experiment_manager.run_experiments()
        print("Fold {fold}: Experiment completed.")
