# mlflow_utils/model_transition.py

import mlflow
from .model_selector import get_best_model
from loguru import logger
from typing import Optional

def transition_model(
    experiment_name: str,
    model_name: str,
    stage: str,
    metric: str = "accuracy",
    version: Optional[int] = None
):
    """
    Transition a model to a specified stage within MLflow based on performance metrics or version.

    This function manages the transition of a model within MLflow by either:
    - Registering the best model based on a specified metric and transitioning it to the desired stage.
    - Transitioning a specific version of an already registered model to the desired stage.

    Args:
        experiment_name (str):
            The name of the MLflow experiment from which to retrieve the best model.
        model_name (str):
            The registered name of the model in MLflow.
        stage (str):
            The target stage to transition the model to (e.g., "Staging", "Production", "Archived").
        metric (str, optional):
            The performance metric to evaluate and select the best model. Defaults to "accuracy".
        version (Optional[int], optional):
            The specific version number of the model to transition. If `None`, the best model based on the specified metric will be selected and registered. Defaults to `None`.

    Returns:
        None

    Raises:
        ValueError:
            If no best model is found when `version` is `None`.
        mlflow.exceptions.RestException:
            If there is an issue fetching the specified model version from MLflow.
        mlflow.exceptions.MlflowException:
            If there is an issue during model registration or stage transition.

    Example:
        >>> # Transition the best model based on accuracy to Production
        >>> transition_model(
        ...     experiment_name="NLI_Experiment",
        ...     model_name="bert-finetuned-nli",
        ...     stage="Production",
        ...     metric="accuracy"
        ... )
        
        >>> # Transition a specific model version to Staging
        >>> transition_model(
        ...     experiment_name="NLI_Experiment",
        ...     model_name="bert-finetuned-nli",
        ...     stage="Staging",
        ...     version=3
        ... )
    """
    if version is None:
        best_run = get_best_model(experiment_name, metric)
        if not best_run:
            logger.error("No best model found to transition.")
            return
        version = _register_model(model_name, best_run)
    else:
        # Fetch specific version
        model_version = _fetch_model_version(model_name, version)
        if not model_version:
            logger.error(f"Model version {version} does not exist.")
            return
        _update_stage(model_version, stage)
    logger.info(f"Model {model_name} transitioned to stage: {stage}")

def _register_model( model_name, run ) -> int:
    """
    Register a model from a specific MLflow run.

    This helper function registers a model artifact from the given MLflow run into the MLflow Model Registry under the specified model name. It returns the version number assigned to the newly registered model.

    Args:
        model_name (str):
            The name under which the model will be registered in the MLflow Model Registry.
        run (mlflow.entities.Run):
            The MLflow Run object from which to retrieve the model artifact.

    Returns:
        int:
            The version number of the registered model in the MLflow Model Registry.

    Raises:
        mlflow.exceptions.MlflowException:
            If there is an issue during the model registration process.

    Example:
        >>> from mlflow.entities import Run
        >>> run = mlflow.get_run("1234567890abcdef")
        >>> version = _register_model("bert-finetuned-nli", run)
        >>> print(f"Registered model version: {version}")
        Registered model version: 1
    """
    # Ensure you have set the model name somewhere in your class
    model_uri = f"runs:/{run.info.run_id}/model"
    print(model_uri)
    logger.info(f"Registering model from run: {run.info.run_id}")

    # Use mlflow.register_model instead of client.register_model
    model_version = mlflow.register_model(model_uri, model_name)
    
    logger.info(f"Model registered with version: {model_version.version}")
    return int(model_version.version)

def _fetch_model_version( model_name, version: int, ):
    """
    Fetch a specific version of a registered model from MLflow.

    This helper function retrieves a particular version of a model from the MLflow Model Registry. If the specified version does not exist, it logs an error and returns `None`.

    Args:
        model_name (str):
            The registered name of the model in MLflow.
        version (int):
            The version number of the model to retrieve.

    Returns:
        Optional[mlflow.entities.ModelVersion]:
            The MLflow ModelVersion object if found, else `None`.

    Raises:
        mlflow.exceptions.MlflowException:
            If there is an issue accessing the model version from MLflow.

    Example:
        >>> model_version = _fetch_model_version("bert-finetuned-nli", 2)
        >>> if model_version:
        ...     print(f"Fetched model version: {model_version.version}")
        ... else:
        ...     print("Model version not found.")
        Fetched model version: 2
    """
    client = mlflow.tracking.MlflowClient()
    try:
        model_version = client.get_model_version(model_name, str(version))
        return model_version
    except mlflow.exceptions.RestException as e:
        logger.error(f"Error fetching model version {version}: {e}")
        return None

def _update_stage( model_name, model_version, stage: str):
    """
    Update the stage of a specific model version in MLflow.

    This helper function transitions the specified version of a registered model to the desired stage in the MLflow Model Registry. It also archives existing versions in that stage to maintain stage integrity.

    Args:
        model_name (str):
            The registered name of the model in MLflow.
        model_version (mlflow.entities.ModelVersion):
            The specific version of the model to transition.
        stage (str):
            The target stage to transition the model to (e.g., "Staging", "Production", "Archived").

    Returns:
        None

    Raises:
        mlflow.exceptions.MlflowException:
            If there is an issue during the stage transition process.

    Example:
        >>> model_version = _fetch_model_version("bert-finetuned-nli", 2)
        >>> if model_version:
        ...     _update_stage("bert-finetuned-nli", model_version, "Production")
        ...     print("Model stage updated successfully.")
        ... else:
        ...     print("Model version not found.")
        Model stage updated successfully.
    """
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage=stage,
        archive_existing_versions=True
    )
    logger.info(f"Model version {model_version.version} transitioned to {stage}")
