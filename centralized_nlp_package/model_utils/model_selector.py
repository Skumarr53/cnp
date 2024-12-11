# mlflow_utils/model_selector.py

import mlflow
from typing import Optional, List, Dict, Any
from loguru import logger

class ModelSelector:
    def __init__(self, experiment_name: str, metric: str = "accuracy"):
        self.experiment_name = experiment_name
        self.metric = metric
        self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not self.experiment:
            logger.error(f"Experiment {self.experiment_name} does not exist.")
            raise ValueError(f"Experiment {self.experiment_name} does not exist.")
        logger.info(f"ModelSelector initialized for experiment: {self.experiment_name}")

    def get_best_model(self) -> Optional[mlflow.entities.Run]:
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{self.metric} DESC"],
            max_results=1
        )
        if runs.empty:
            logger.warning("No runs found in the experiment.")
            return None
        best_run_id = runs.iloc[0]["run_id"]
        best_run = mlflow.get_run(best_run_id)
        logger.info(f"Best run ID: {best_run_id} with {self.metric}: {runs.iloc[0][f'metrics.{self.metric}']}")
        return best_run
    
    def get_best_models_by_tag(self, tag) -> List[mlflow.entities.Run]:
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{self.metric} DESC"]
        )
    
        if runs.empty:
            logger.warning("No runs found in the experiment.")
            return []

        # Group runs by the specified tag
        grouped_runs = runs.groupby(f"tags.{tag}")

        best_runs = []
        for group_name, group in grouped_runs:
            # Sort the group by the metric in descending order and pick the best run
            best_run_row = group.sort_values(by=f"metrics.{self.metric}", ascending=False).iloc[0]
            best_run_id = best_run_row["run_id"]
            best_run = mlflow.get_run(best_run_id)
            logger.info(f"Best run for {tag}={group_name}: Run ID {best_run_id} with {self.metric}: {best_run_row[f'metrics.{self.metric}']}")
            best_runs.append(best_run)

        return best_runs

    def get_best_models_by_param(self, param) -> List[mlflow.entities.Run]:
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{self.metric} DESC"]
        )
        
        if runs.empty:
            logger.warning("No runs found in the experiment.")
            return []

        # Group runs by the specified parameter
        grouped_runs = runs.groupby(f"params.{param}")

        best_runs = []
        for group_name, group in grouped_runs:
            # Sort the group by the metric in descending order and pick the best run
            best_run_row = group.sort_values(by=f"metrics.{self.metric}", ascending=False).iloc[0]
            best_run_id = best_run_row["run_id"]
            best_run = mlflow.get_run(best_run_id)
            logger.info(f"Best run for {param}={group_name}: Run ID {best_run_id} with {self.metric}: {best_run_row[f'metrics.{self.metric}']}")
            best_runs.append(best_run)

        return best_runs

    def list_available_models(self) -> List[Dict[str, Any]]:
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{self.metric} DESC"]
        )
        models = []
        for _, row in runs.iterrows():
            models.append({
                "run_id": row["run_id"],
                "run_name": row["tags.mlflow.runName"],
                "metrics": {key.replace("metrics.", ""): value for key, value in row.items() if key.startswith("metrics.")}
            })
        logger.info(f"Total models found: {len(models)}")
        return models
