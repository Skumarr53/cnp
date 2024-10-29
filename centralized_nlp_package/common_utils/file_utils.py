# centralized_nlp_package/common_utils/file_utils.py
import os
from loguru import logger
from centralized_nlp_package.utils import FilesNotLoadedException

def load_content_from_txt(file_path: str) -> str:
    """
    Reads the entire content of a text file from the given file path.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The content of the text file.

    Raises:
        FilesNotLoadedException: If the file is not found at the given path.

    Example:
        >>> from centralized_nlp_package.common_utils import load_content_from_txt
        >>> content = load_content_from_txt("data/sample.txt")
        >>> print(content)
        'This is a sample text file.'
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        logger.debug(f"Loaded content from {file_path}.")
        return content
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise FilesNotLoadedException(f"File not found: {file_path}") from e
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        raise FilesNotLoadedException(f"Error loading file {file_path}: {e}") from e


