# mlflow_utils/model_transition.py

import mlflow
from .model_selector import ModelSelector
from loguru import logger
from typing import Optional

class ModelTransition:
    def __init__(self, model_name: str, registry_uri: str = "models"):
        self.model_name = model_name
        self.registry_uri = registry_uri
        mlflow.set_registry_uri(self.registry_uri)
        logger.info(f"ModelTransition initialized for model: {self.model_name}")

    def transition_model(
        self,
        stage: str,
        experiment_name: str,
        metric: str = "accuracy",
        version: Optional[int] = None
    ):
        selector = ModelSelector(experiment_name, metric)
        if version is None:
            best_run = selector.get_best_model()
            if not best_run:
                logger.error("No best model found to transition.")
                return
            version = self._register_model(best_run)
        else:
            # Fetch specific version
            model_version = self._fetch_model_version(version)
            if not model_version:
                logger.error(f"Model version {version} does not exist.")
                return
            self._update_stage(model_version, stage)
        logger.info(f"Model {self.model_name} transitioned to stage: {stage}")

    def _register_model(self, run) -> int:
        # Ensure you have set the model name somewhere in your class
        model_uri = f"runs:/{run.info.run_id}/model"
        print(model_uri)
        logger.info(f"Registering model from run: {run.info.run_id}")

        # Use mlflow.register_model instead of client.register_model
        model_version = mlflow.register_model(model_uri, self.model_name)
        
        logger.info(f"Model registered with version: {model_version.version}")
        return int(model_version.version)

    def _fetch_model_version(self, version: int):
        client = mlflow.tracking.MlflowClient()
        try:
            model_version = client.get_model_version(self.model_name, str(version))
            return model_version
        except mlflow.exceptions.RestException as e:
            logger.error(f"Error fetching model version {version}: {e}")
            return None

    def _update_stage(self, model_version, stage: str):
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=self.model_name,
            version=model_version.version,
            stage=stage,
            archive_existing_versions=True
        )
        logger.info(f"Model version {model_version.version} transitioned to {stage}")
