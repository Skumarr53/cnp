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
    # Ensure you have set the model name somewhere in your class
    model_uri = f"runs:/{run.info.run_id}/model"
    print(model_uri)
    logger.info(f"Registering model from run: {run.info.run_id}")

    # Use mlflow.register_model instead of client.register_model
    model_version = mlflow.register_model(model_uri, model_name)
    
    logger.info(f"Model registered with version: {model_version.version}")
    return int(model_version.version)

def _fetch_model_version( model_name, version: int, ):
    client = mlflow.tracking.MlflowClient()
    try:
        model_version = client.get_model_version(model_name, str(version))
        return model_version
    except mlflow.exceptions.RestException as e:
        logger.error(f"Error fetching model version {version}: {e}")
        return None

def _update_stage( model_name, model_version, stage: str):
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage=stage,
        archive_existing_versions=True
    )
    logger.info(f"Model version {model_version.version} transitioned to {stage}")
