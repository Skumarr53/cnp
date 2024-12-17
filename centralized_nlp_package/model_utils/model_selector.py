# mlflow_utils/model_selector.py

import mlflow
from typing import Optional, List, Dict, Any
from loguru import logger

def get_best_model(experiment_name: str, metric: str = "accuracy") -> Optional[mlflow.entities.Run]:
    runs = mlflow.search_runs(
        experiment_ids=[experiment_name.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )
    if runs.empty:
        logger.warning("No runs found in the experiment.")
        return None
    best_run_id = runs.iloc[0]["run_id"]
    best_run = mlflow.get_run(best_run_id)
    logger.info(f"Best run ID: {best_run_id} with {metric}: {runs.iloc[0][f'metrics.{metric}']}")
    return best_run

def get_best_models_by_tag(experiment_name, tag, metric = "accuracy") -> List[mlflow.entities.Run]:
    runs = mlflow.search_runs(
        experiment_ids=[experiment_name.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )

    if runs.empty:
        logger.warning("No runs found in the experiment.")
        return []

    # Group runs by the specified tag
    grouped_runs = runs.groupby(f"tags.{tag}")

    best_runs = []
    for group_name, group in grouped_runs:
        # Sort the group by the metric in descending order and pick the best run
        best_run_row = group.sort_values(by=f"metrics.{metric}", ascending=False).iloc[0]
        best_run_id = best_run_row["run_id"]
        best_run = mlflow.get_run(best_run_id)
        logger.info(f"Best run for {tag}={group_name}: Run ID {best_run_id} with {metric}: {best_run_row[f'metrics.{metric}']}")
        best_runs.append(best_run)

    return best_runs

def get_best_models_by_param(experiment_name, param, metric = "accuracy") -> List[mlflow.entities.Run]:
    runs = mlflow.search_runs(
        experiment_ids=[experiment_name.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )
    
    if runs.empty:
        logger.warning("No runs found in the experiment.")
        return []

    # Group runs by the specified parameter
    grouped_runs = runs.groupby(f"params.{param}")

    best_runs = []
    for group_name, group in grouped_runs:
        # Sort the group by the metric in descending order and pick the best run
        best_run_row = group.sort_values(by=f"metrics.{metric}", ascending=False).iloc[0]
        best_run_id = best_run_row["run_id"]
        best_run = mlflow.get_run(best_run_id)
        logger.info(f"Best run for {param}={group_name}: Run ID {best_run_id} with {metric}: {best_run_row[f'metrics.{metric}']}")
        best_runs.append(best_run)

    return best_runs

def list_available_models(experiment_name, metric = "accuracy") -> List[Dict[str, Any]]:
    runs = mlflow.search_runs(
        experiment_ids=[experiment_name.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
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
