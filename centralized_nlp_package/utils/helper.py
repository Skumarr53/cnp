from loguru import logger
from typing import Any, Tuple, Union, List, Callable, Optional

def determine_environment(provided_env: Optional[str] = None) -> str:
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
        if env not in ['dev', 'stg', 'prod']:
            logger.error(f"Invalid environment provided: {provided_env}. Must be one of 'dev', 'stg', 'prod'.")
            raise ValueError(f"Invalid environment: {provided_env}. Choose from 'dev', 'stg', 'prod'.")
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
        env = 'dev'
    elif workspace_name.startswith(stg_workspace_name):
        env = 'stg'
    elif workspace_name.startswith(prod_workspace_name):
        env = 'prod'
    else:
        logger.error(f"Workspace name '{workspace_name}' does not match any known environments.")
        raise ValueError(f"Unknown workspace name: {workspace_name}. Cannot determine environment.")
    
    logger.info(f"Environment auto-detected based on workspace name: {env}")
    return env