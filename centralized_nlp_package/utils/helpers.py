# centralized_nlp_package/utils/helpers.py
import os
import re
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple, Union
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from dateutil.relativedelta import relativedelta
from centralized_nlp_package import config



def load_file(file_path: str) -> str:
    """
    Loads a SQL query from an external .sql file.
    
    Args:
        file_path (str): Path to the SQL file.
    
    Returns:
        str: The SQL query as a string.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError as e:
        logger.error(f"Query file not found: {file_path}")
        raise FileNotFoundError(f"Query file not found: {file_path}") from e


def get_date_range(years_back: int = 0, months_back: int = 0) -> Tuple[str, str]:
    """
    Calculate the date range for the query based on months or years.

    Args:
        months_back (int): The number of months to go back from the current date (default is 0).
        years_back (int): The number of years to go back from the current year (default is 0).

    Returns:
        Tuple[str, str]: A tuple containing the minimum and maximum dates in the format 'YYYY-MM-DD'.
    """
    end_date = datetime.now()
    
    # Calculate start date based on months_back
    if months_back > 0:
        start_date = end_date - relativedelta(months=months_back)
    else:
        start_date = end_date

    # Calculate start date based on years_back
    if years_back > 0:
        start_year = end_date.year - years_back
        min_date = f"{start_year}-{end_date.month:02d}-01"
    else:
        min_date = f"{start_date.year}-{start_date.month:02d}-01"

    # Max date is always the start of the current month
    max_date = f"{end_date.year}-{end_date.month:02d}-01"

    return min_date, max_date


def format_date(date: datetime) -> str:
    """
    Formats a datetime object to 'YYYY-MM-DD' string.

    Args:
        date (datetime): The date to format.

    Returns:
        str: Formatted date string.
    """
    formatted_date = date.strftime('%Y-%m-%d')
    logger.debug(f"Formatted date: {formatted_date}")
    return formatted_date

def format_string_template(template: str, **kwargs) -> Path:
    """
    Constructs the formated string using the provided template and dynamic components.

    Args:
        template (str): Template string with placeholders.
        **kwargs: Variable number of keyword arguments to replace placeholders in the template.

    Returns:
        abs_string: Constructed abosulte string.
    """
    # Find placeholders in the template (in the form {key})
    placeholders = re.findall(r"{(.*?)}", template)

    if not all(placeholder in kwargs for placeholder in placeholders):
        logger.error(f"Template placeholders, not matches with parameters provided.")
        raise ValueError(f"Template placeholders do not match the provided parameters.")
    
    # Construct the path string
    format_string = template.format(**kwargs)
    return format_string


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
        FileNotFoundError: If the provided path to the query file does not exist.
    """
    # Determine if the input is a file path or a query string
    if os.path.isfile(query_input):
        # Load the query string from the provided file path
        base_query = load_file(query_input)
    else:
        # Use the provided string as the query
        base_query = query_input

    # Find placeholders in the query string (in the form {param})
    logger.debug("contructing formatted query")
    final_query = format_string_template(base_query, **kwargs)

    return final_query


def df_remove_rows_with_keywords(df: pd.DataFrame, column_name: str, keywords: list) -> pd.DataFrame:
    """
    Filters a DataFrame by removing rows where the specified column contains any of the keywords.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        column_name (str): The name of the column to check for keywords.
        keywords (list): A list of strings or keywords to filter out.

    Returns:
        pd.DataFrame: The filtered DataFrame.

    Raises:
        Warning: If any keyword is not found in the specified column.
    """
    # Check if all keywords are present in the column
    missing_keywords = [keyword for keyword in keywords if not df[column_name].str.contains(keyword, na=False).any()]
    if missing_keywords:
        logger.warning(f"The following keywords were not found in the column '{column_name}': {missing_keywords}")

    # Filter out rows containing any of the keywords
    mask = df[column_name].apply(lambda x: not any(keyword == str(x) for keyword in keywords))
    logger.debug("contructing formatted query")
    filtered_df = df[mask]

    return filtered_df