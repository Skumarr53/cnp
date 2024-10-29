# centralized_nlp_package/common_utils/string_utils.py
import os
import re
from typing import Union
from loguru import logger
from centralized_nlp_package.common_utils import load_content_from_txt



def format_string_template(template: str, **kwargs) -> str:
    """
    Constructs a formatted string by replacing placeholders in the template with provided keyword arguments.

    Args:
        template (str): Template string containing placeholders in the form {key}.
        **kwargs: Variable keyword arguments to replace placeholders.

    Returns:
        str: The formatted string with placeholders replaced.

    Raises:
        ValueError: If any placeholder in the template does not have a corresponding keyword argument.

    Example:
        >>> from centralized_nlp_package.common_utils import format_string_template
        >>> template = "Hello, {name}! Today is {day}."
        >>> format_string_template(template, name="Alice", day="Monday")
        'Hello, Alice! Today is Monday.'
    """
    placeholders = re.findall(r"{(.*?)}", template)

    missing_keys = [key for key in placeholders if key not in kwargs]
    if missing_keys:
        logger.error(f"Missing keys for placeholders: {missing_keys}")
        raise ValueError(f"Missing keys for placeholders: {missing_keys}")

    try:
        format_string = template.format(**kwargs)
        logger.debug(f"Formatted string: {format_string}")
        return format_string
    except KeyError as e:
        logger.error(f"Missing key during string formatting: {e}")
        raise ValueError(f"Missing key during string formatting: {e}") from e
    except Exception as e:
        logger.error(f"Error formatting string: {e}")
        raise ValueError(f"Error formatting string: {e}") from e


def query_constructor(query_input: Union[str, os.PathLike], **kwargs) -> str:
    """
    Constructs a query string by loading from a file or using a provided string and replacing placeholders with provided parameters.

    Args:
        query_input (Union[str, os.PathLike]): The file path to the query or the query string itself.
        **kwargs: Parameters to replace placeholders in the query.

    Returns:
        str: The constructed query string with parameters substituted.

    Raises:
        ValueError: If placeholders in the query do not match the provided parameters.
        FilesNotLoadedException: If the provided path to the query file does not exist.

    Example:
        >>> from centralized_nlp_package.common_utils import query_constructor
        >>> template = "SELECT * FROM users WHERE signup_date > '{start_date}' AND signup_date < '{end_date}';"
        >>> query = query_constructor(template, start_date="2022-01-01", end_date="2022-12-31")
        >>> print(query)
        "SELECT * FROM users WHERE signup_date > '2022-01-01' AND signup_date < '2022-12-31';"
    """
    if os.path.isfile(query_input):
        base_query = load_content_from_txt(query_input)
        logger.debug(f"Loaded query from file: {query_input}")
    else:
        base_query = str(query_input)
        logger.debug("Using provided query string.")

    final_query = format_string_template(base_query, **kwargs)

    return final_query