# centralized_nlp_package/utils/logging_setup.py

from loguru import logger
from pathlib import Path
import os
import sys


def setup_logging(log_file_path: str = "logs/log_file.log", env: str = "dev") -> None:
    """
    Sets up the Loguru logger with both console and file handlers based on the environment.

    Args:
        log_file_path (str, optional): Path to the log file. Defaults to "logs/log_file.log".
        env (str, optional): Environment type to determine logging level. 
                             Use "prod" for production (ERROR level) and "dev" for development (DEBUG level). Defaults to "dev".

    Example:
        >>> from centralized_nlp_package.utils import setup_logging
        >>> setup_logging(log_file_path="app_logs/app.log", env="prod")
        >>> logger.info("This is an info message.")
        >>> logger.error("This is an error message.")
    """
    # Remove any existing handlers (useful when setting up logging multiple times in tests or notebooks)
    logger.remove()

    # Determine logging level based on environment
    if env.lower() == "prod":
        log_level = "ERROR"
    else:
        log_level = "DEBUG"

    # Ensure log directory exists
    log_directory = Path(log_file_path).parent
    os.makedirs(log_directory, exist_ok=True)

    # Console Handler
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:7}</level> | "
            "<cyan>{file}:{line}</cyan> | "
            "{message}"
        ),
        diagnose=True,  # Includes detailed information about where the log is coming from
        backtrace=True,  # Provides a backtrace for error messages
        enqueue=True,    # Makes the logging calls thread-safe
    )

    # File Handler with Rotation and Retention
    logger.add(
        log_file_path,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {message}",
        rotation="10 MB",      # Rotate log file after it reaches 10 MB
        retention="7 days",    # Keep log files for 7 days
        compression="zip",     # Compress rotated log files
    )

    logger.info("Logging setup completed.")
