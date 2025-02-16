# mlflow_utils/model_selector.py

import mlflow
from typing import Optional, List, Dict, Any
#from loguru import logger

def get_best_model(experiment_name: str, metric: str = "accuracy") -> Optional[mlflow.entities.Run]:
    runs = mlflow.search_runs(
        experiment_ids=[experiment_name.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )
    """
    Retrieve the best run from an MLflow experiment based on a specified metric.

    This function searches through all runs in the given MLflow experiment and identifies the run with the highest value for the specified metric. If no runs are found, it logs a warning and returns `None`.

    Args:
        experiment_name (str):
            The name of the MLflow experiment from which to retrieve runs.
        metric (str, optional):
            The metric to evaluate runs by. The function will select the run with the highest value for this metric. Defaults to "accuracy".

    Returns:
        Optional[mlflow.entities.Run]:
            The MLflow Run object corresponding to the best run based on the specified metric. Returns `None` if no runs are found.

    Raises:
        mlflow.exceptions.MlflowException:
            If there is an issue accessing the MLflow experiment or runs.

    Example:
        >>> best_run = get_best_model("NLI_Experiment", metric="f1_score")
        >>> if best_run:
        ...     print(f"Best run ID: {best_run.info.run_id}")
        ...     print(f"Best {metric}: {best_run.data.metrics[metric]}")
        ... else:
        ...     print("No runs found in the experiment.")
    """
    if runs.empty:
        print("No runs found in the experiment.")
        return None
    best_run_id = runs.iloc[0]["run_id"]
    best_run = mlflow.get_run(best_run_id)
    print("Best run ID: {best_run_id} with {metric}: {runs.iloc[0][f'metrics.{metric}']}")
    return best_run

def get_best_models_by_tag(experiment_name, tag, metric = "accuracy") -> List[mlflow.entities.Run]:
    """
    Retrieve the best runs for each unique value of a specified tag within an MLflow experiment based on a specified metric.

    This function groups runs in the given MLflow experiment by the specified tag and selects the best run within each group based on the highest value of the specified metric. If no runs are found, it logs a warning and returns an empty list.

    Args:
        experiment_name (str):
            The name of the MLflow experiment from which to retrieve runs.
        tag (str):
            The tag key to group runs by. Each unique value of this tag will have its best run selected.
        metric (str, optional):
            The metric to evaluate runs by within each tag group. The function will select the run with the highest value for this metric. Defaults to "accuracy".

    Returns:
        List[mlflow.entities.Run]:
            A list of MLflow Run objects, each representing the best run for a unique tag value based on the specified metric.

    Raises:
        mlflow.exceptions.MlflowException:
            If there is an issue accessing the MLflow experiment or runs.

    Example:
        >>> best_runs = get_best_models_by_tag("NLI_Experiment", tag="model_version", metric="f1_score")
        >>> for run in best_runs:
        ...     print(f"Run ID: {run.info.run_id}, Model Version: {run.data.tags['model_version']}, F1 Score: {run.data.metrics['f1_score']}")
    """
    runs = mlflow.search_runs(
        experiment_ids=[experiment_name.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )

    if runs.empty:
        print("No runs found in the experiment.")
        return []

    # Group runs by the specified tag
    grouped_runs = runs.groupby(f"tags.{tag}")

    best_runs = []
    for group_name, group in grouped_runs:
        # Sort the group by the metric in descending order and pick the best run
        best_run_row = group.sort_values(by=f"metrics.{metric}", ascending=False).iloc[0]
        best_run_id = best_run_row["run_id"]
        best_run = mlflow.get_run(best_run_id)
        print("Best run for {tag}={group_name}: Run ID {best_run_id} with {metric}: {best_run_row[f'metrics.{metric}']}")
        best_runs.append(best_run)

    return best_runs

def get_best_models_by_param(experiment_name, param, metric = "accuracy") -> List[mlflow.entities.Run]:
    """
    Retrieve the best runs for each unique value of a specified parameter within an MLflow experiment based on a specified metric.

    This function groups runs in the given MLflow experiment by the specified parameter and selects the best run within each group based on the highest value of the specified metric. If no runs are found, it logs a warning and returns an empty list.

    Args:
        experiment_name (str):
            The name of the MLflow experiment from which to retrieve runs.
        param (str):
            The parameter key to group runs by. Each unique value of this parameter will have its best run selected.
        metric (str, optional):
            The metric to evaluate runs by within each parameter group. The function will select the run with the highest value for this metric. Defaults to "accuracy".

    Returns:
        List[mlflow.entities.Run]:
            A list of MLflow Run objects, each representing the best run for a unique parameter value based on the specified metric.

    Raises:
        mlflow.exceptions.MlflowException:
            If there is an issue accessing the MLflow experiment or runs.

    Example:
        >>> best_runs = get_best_models_by_param("NLI_Experiment", param="learning_rate", metric="f1_score")
        >>> for run in best_runs:
        ...     print(f"Run ID: {run.info.run_id}, Learning Rate: {run.data.params['learning_rate']}, F1 Score: {run.data.metrics['f1_score']}")
    """
    runs = mlflow.search_runs(
        experiment_ids=[experiment_name.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )
    
    if runs.empty:
        print("No runs found in the experiment.")
        return []

    # Group runs by the specified parameter
    grouped_runs = runs.groupby(f"params.{param}")

    best_runs = []
    for group_name, group in grouped_runs:
        # Sort the group by the metric in descending order and pick the best run
        best_run_row = group.sort_values(by=f"metrics.{metric}", ascending=False).iloc[0]
        best_run_id = best_run_row["run_id"]
        best_run = mlflow.get_run(best_run_id)
        print("Best run for {param}={group_name}: Run ID {best_run_id} with {metric}: {best_run_row[f'metrics.{metric}']}")
        best_runs.append(best_run)

    return best_runs

def list_available_models(experiment_name, metric = "accuracy"):
    """
    List all available models from an MLflow experiment, sorted by a specified metric in descending order.

    This function retrieves all runs from the given MLflow experiment, sorts them based on the specified metric in descending order, and compiles a list of dictionaries containing run details such as run ID, run name, and metrics.

    Args:
        experiment_name (str):
            The name of the MLflow experiment from which to list models.
        metric (str, optional):
            The metric to sort the models by. Models with higher values for this metric will appear first. Defaults to "accuracy".

    Returns:
        List[Dict[str, Any]]:
            A list of dictionaries, each representing a model with keys:
                - "run_id" (str): The unique identifier of the MLflow run.
                - "run_name" (str): The name of the MLflow run.
                - "metrics" (Dict[str, Any]): A dictionary of metric names and their corresponding values for the run.

    Raises:
        mlflow.exceptions.MlflowException:
            If there is an issue accessing the MLflow experiment or runs.

    Example:
        >>> models = list_available_models("NLI_Experiment", metric="f1_score")
        >>> for model in models:
        ...     print(f"Run ID: {model['run_id']}, Run Name: {model['run_name']}, F1 Score: {model['metrics']['f1_score']}")
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )
    models = []
    for _, row in runs.iterrows():
        models.append({
            "run_id": row["run_id"],
            "run_name": row["tags.mlflow.runName"],
            "metrics": {key.replace("metrics.", ""): value for key, value in row.items() if key.startswith("metrics.")}
        })
    print("Total models found: {len(models)}")
    return models

