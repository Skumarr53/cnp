
from loguru import logger
import os
from omegaconf import DictConfig
import hydra
from typing import Optional

def determine_environment(provided_env: Optional[str] = 'quant') -> str:
    """
    Determines the environment based on the provided argument or auto-detects it using the DataBricks workspace name.
    
    Args:
        provided_env (Optional[str]): The environment specified by the user ('dev', 'stg', 'prod').
                                      If not provided, the environment will be auto-detected.
    
    Returns:
        str: The determined environment ('dev', 'stg', 'prod').
    
    Raises:
        ValueError: If the environment cannot be determined.
    """
    if provided_env:
        env = provided_env.lower()
        if env not in ['quant', 'quant_stg', 'quant_live']:
            logger.error(f"Invalid environment provided: {provided_env}. Must be one of 'quant', 'quant_stg', 'quant_live'.")
            raise ValueError(f"Invalid environment: {provided_env}. Choose from 'quant', 'quant_stg', 'quant_live'.")
        logger.info(f"Environment provided by user: {env}")
        return env
    
    try:
        # Retrieve the notebook path using dbutils (Databricks)
        workspace_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        logger.debug(f"Retrieved workspace name from notebook path: {workspace_name}")
    except Exception as e:
        logger.error(f"Error retrieving workspace name: {e}")
        raise ValueError("Unable to determine the workspace name for environment detection.") from e
    
    #TODO: replace with actual 
    # Define your workspace names
    dev_workspace_name = "/Users/dev_user/dev_workspace"
    stg_workspace_name = "/Users/stg_user/stg_workspace"
    prod_workspace_name = "/Users/prod_user/prod_workspace"
    
    if workspace_name == '2762743938046900':
        env = 'quant'
    elif workspace_name.startswith(stg_workspace_name):
        env = 'quant_stg'
    elif workspace_name.startswith(prod_workspace_name):
        env = 'quant_live'
    else:
        logger.error(f"Workspace name '{workspace_name}' does not match any known environments.")
        raise ValueError(f"Unknown workspace name: {workspace_name}. Cannot determine environment.")
    
    logger.info(f"Environment auto-detected based on workspace name: {env}")
    return env



def load_config_from_file(file_path: str) -> DictConfig:
    """
    Load a configuration file using Hydra.

    This function takes a full file path to a configuration file, extracts the directory
    and filename, and uses Hydra to initialize and compose the configuration.

    Args:
        file_path (str): The full path to the configuration file (e.g., '/path/to/config.yaml').

    Returns:
        DictConfig: The loaded configuration as a DictConfig object.

    Raises:
        RuntimeError: If there is an error loading the configuration, including issues with
                      the file path or the contents of the configuration file.
    
    Example:
        config = get_config("/path/to/your/config.yaml")
    """
    try:
        # Extract directory and filename from the full file path
        config_dir = os.path.dirname(file_path)
        config_name = os.path.splitext(os.path.basename(file_path))[0]

        # Initialize Hydra with the specified config directory
        with hydra.initialize(config_path=config_dir):
            _config = hydra.compose(config_name=config_name)
    
    except Exception as e:
        raise RuntimeError(f"Error loading configuration: {e}")

    return _config