from dask.distributed import Client
from dask.diagnostics import ProgressBar

def initialize_dask_client(n_workers: int, threads_per_worker: int = 1) -> Client:
    """
    Initializes a Dask client with specified workers and threads.
    
    Args:
        n_workers (int): Number of workers to use.
        threads_per_worker (int): Number of threads per worker.

    Returns:
        Client: An instance of the Dask distributed client.
    """
    ## TODO: add params to config
    client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker)
    return client

def dask_compute_with_progress(dask_dataframe, use_progress: bool = True):
    """
    Computes a Dask DataFrame optionally displaying a progress bar.
    
    Args:
        dask_dataframe: The Dask DataFrame to be computed.
        use_progress (bool): Whether to use a progress bar for monitoring.

    Returns:
        Computed DataFrame: The computed Dask DataFrame.
    """
    if use_progress:
        with ProgressBar():
            return dask_dataframe.compute()
    else:
        return dask_dataframe.compute()
