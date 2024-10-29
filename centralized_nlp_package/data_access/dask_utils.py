# centralized_nlp_package/data_access/dask_utils.py

from dask.distributed import Client
from dask.diagnostics import ProgressBar
from typing import Any, Optional
import pandas as pd
from loguru import logger
from centralized_nlp_package import config


def initialize_dask_client(n_workers: int = 32, threads_per_worker: int = 1) -> Client:
    """
    Initializes a Dask client with specified number of workers and threads per worker.
    
    Args:
        n_workers (int): Number of workers to initialize.
        threads_per_worker (int, optional): Number of threads per worker. Defaults to 1.
    
    Returns:
        Client: An instance of the Dask distributed client.
    
    Example:
        >>> from centralized_nlp_package.data_access import initialize_dask_client
        >>> client = initialize_dask_client(n_workers=32, threads_per_worker=1)
        >>> print(client)
        <distributed.client.Client object at 0x...>
    """
    # load default config for yaml
    try:
        logger.info(
            f"Initializing Dask client with {n_workers} workers and "
            f"{threads_per_worker} threads per worker."
        )
        client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker)
        logger.info("Dask client initialized successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Dask client: {e}")
        raise


def dask_compute_with_progress(dask_dataframe: Any, use_progress: bool = True) -> Any:
    """
    Computes a Dask DataFrame, optionally displaying a progress bar.
    
    Args:
        dask_dataframe (Any): The Dask DataFrame to be computed.
        use_progress (bool, optional): Whether to display a progress bar during computation. Defaults to True.
    
    Returns:
        Any: The computed DataFrame (e.g., pandas DataFrame).
    
    Example:
        >>> >>> from centralized_nlp_package.data_access import dask_compute_with_progress
        >>> import dask.dataframe as dd
        >>> df = dd.read_csv('data/*.csv')
        >>> computed_df = dask_compute_with_progress(df, use_progress=True)
        >>> print(computed_df.head())
           column1  column2
        0        1        4
        1        2        5
        2        3        6
    """
    try:
        if use_progress:
            logger.info("Starting computation with progress bar.")
            with ProgressBar():
                result = dask_dataframe.compute()
        else:
            logger.info("Starting computation without progress bar.")
            result = dask_dataframe.compute()
        logger.info("Dask DataFrame computed successfully.")
        return result
    except Exception as e:
        logger.error(f"Failed to compute Dask DataFrame: {e}")
        raise
