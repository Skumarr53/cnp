# centralized_nlp_package/utils/helpers.py
import os
import re
import ast
from typing import Any, Tuple, Union, List, Callable, Optional

import dask.dataframe as dd
import pandas as pd
from datetime import datetime
from pathlib import Path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from dateutil.relativedelta import relativedelta
from centralized_nlp_package import config
from centralized_nlp_package.utils import FilesNotLoadedException



def load_file(file_path: str) -> str:
    """
    Loads the content from an external file (e.g., SQL query) as a string.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: The content of the file as a string.

    Raises:
        FilesNotLoadedException: If the file is not found.

    Example:
        >>> query = load_file("queries/get_users.sql")
        >>> print(query)
        "SELECT * FROM users;"
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



def get_date_range(years_back: int = 0, months_back: int = 0) -> Tuple[str, str]:
    """
    Calculates the date range based on the current date minus specified years and/or months.

    Args:
        years_back (int, optional): Number of years to go back from the current date. Defaults to 0.
        months_back (int, optional): Number of months to go back from the current date. Defaults to 0.

    Returns:
        Tuple[str, str]: A tuple containing the start date and end date in 'YYYY-MM-DD' format.

    Example:
        >>> start_date, end_date = get_date_range(years_back=1, months_back=2)
        >>> print(start_date, end_date)
        '2022-08-01' '2023-10-01'
    """
    end_date = datetime.now()
    start_date = end_date - relativedelta(years=years_back, months=months_back)

    min_date = f"{start_date.year}-{start_date.month:02d}-01"
    max_date = f"{end_date.year}-{end_date.month:02d}-01"

    logger.debug(f"Calculated date range from {min_date} to {max_date}.")
    return min_date, max_date



def format_date(date: datetime) -> str:
    """
    Formats a datetime object to a string in 'YYYY-MM-DD' format.

    Args:
        date (datetime): The date to format.

    Returns:
        str: The formatted date string.

    Example:
        >>> from datetime import datetime
        >>> format_date(datetime(2023, 9, 15))
        '2023-09-15'
    """
    formatted_date = date.strftime('%Y-%m-%d')
    logger.debug(f"Formatted date: {formatted_date}")
    return formatted_date

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
        >>> template = "Hello, {name}! Today is {day}."
        >>> format_string_template(template, name="Alice", day="Monday")
        'Hello, Alice! Today is Monday.'
    """
    # Find placeholders in the template (in the form {key})
    placeholders = re.findall(r"{(.*?)}", template)

    missing_keys = [key for key in placeholders if key not in kwargs]
    if missing_keys:
        logger.error(f"Missing keys for placeholders: {missing_keys}")
        raise ValueError(f"Missing keys for placeholders: {missing_keys}")

    # Construct the formatted string
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
        >>> template = "SELECT * FROM users WHERE signup_date > '{start_date}' AND signup_date < '{end_date}';"
        >>> query = query_constructor(template, start_date="2022-01-01", end_date="2022-12-31")
        >>> print(query)
        "SELECT * FROM users WHERE signup_date > '2022-01-01' AND signup_date < '2022-12-31';"
    """
    # Determine if the input is a file path or a query string
    if os.path.isfile(query_input):
        # Load the query string from the provided file path
        base_query = load_file(query_input)
        logger.debug(f"Loaded query from file: {query_input}")
    else:
        # Use the provided string as the query
        base_query = str(query_input)
        logger.debug("Using provided query string.")

    # Replace placeholders in the query string
    final_query = format_string_template(base_query, **kwargs)

    return final_query


def df_remove_rows_with_keywords(
    df: pd.DataFrame, column_name: str, keywords: List[str]
) -> pd.DataFrame:
    """
    Filters a DataFrame by removing rows where the specified column contains any of the keywords.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        column_name (str): The name of the column to check for keywords.
        keywords (List[str]): A list of strings or keywords to filter out.

    Returns:
        pd.DataFrame: The filtered DataFrame.

    Raises:
        ValueError: If the specified column is not found in the DataFrame.

    Example:
        >>> data = {'comments': ["Good product", "Bad service", "Average experience"]}
        >>> df = pd.DataFrame(data)
        >>> filtered_df = df_remove_rows_with_keywords(df, 'comments', ['Bad service'])
        >>> print(filtered_df)
             comments
        0  Good product
        2  Average experience
    """
    # Check if the specified column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"The column '{column_name}' does not exist in the DataFrame.")
    
    # Check if all keywords are present in the column
    missing_keywords = [keyword for keyword in keywords if not df[column_name].str.contains(keyword, na=False).any()]
    if missing_keywords:
        logger.warning(f"The following keywords were not found in the column '{column_name}': {missing_keywords}")

    # Filter out rows containing any of the keywords
    try:
        mask = df[column_name].apply(lambda x: not any(keyword == str(x) for keyword in keywords))
        filtered_df = df[mask]
        logger.info(f"Filtered DataFrame to remove rows containing keywords. Remaining rows: {len(filtered_df)}.")
        return filtered_df
    except Exception as e:
        logger.error(f"An error occurred while filtering the DataFrame: {e}")
        raise


def df_apply_transformations(
    df: Union[pd.DataFrame, dd.DataFrame],
    transformations: List[Tuple[str, Union[str, List[str]], Callable]],
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Applies a set of transformations to a DataFrame based on the given list of transformation tuples.

    Each transformation tuple should contain:
        - new_column (str): The new column name to create or transform.
        - columns_to_use (Union[str, List[str]]): Column(s) to use in the transformation.
        - func (Callable): The transformation function.

    Args:
        df (Union[pd.DataFrame, dd.DataFrame]): The DataFrame to apply transformations on.
        transformations (List[Tuple[str, Union[str, List[str]], Callable]]): 
            A list of tuples where each tuple contains:
                - new_column (str): The new column name to create or transform.
                - columns_to_use (Union[str, List[str]]): Column(s) to use in the transformation.
                - func (Callable): The transformation function.

    Returns:
        Union[pd.DataFrame, dd.DataFrame]: The DataFrame with applied transformations.

    Raises:
        ValueError: If a transformation tuple is invalid.
        Exception: Re-raises any exception that occurs during the transformation process after logging.

    Example:
        >>> def concat_columns(a, b):
        ...     return f"{a}_{b}"
        >>> data = {'col1': ['A', 'B'], 'col2': ['C', 'D']}
        >>> df = pd.DataFrame(data)
        >>> transformations = [
        ...     ('col3', ['col1', 'col2'], lambda row: concat_columns(row['col1'], row['col2']))
        ... ]
        >>> transformed_df = df_apply_transformations(df, transformations)
        >>> print(transformed_df)
          col1 col2 col3
        0    A    C  A_C
        1    B    D  B_D
    """
    for transformation in transformations:
        if len(transformation) != 3:
            logger.error(f"Invalid transformation tuple: {transformation}. Expected 3 elements.")
            raise ValueError(f"Invalid transformation tuple: {transformation}. Expected 3 elements.")

        new_column, columns_to_use, func = transformation

        if not callable(func):
            logger.error(f"Transformation function for column '{new_column}' is not callable.")
            raise ValueError(f"Transformation function for column '{new_column}' is not callable.")

        try:
            if isinstance(columns_to_use, str):
                # Single column transformation
                logger.debug(f"Applying transformation on single column '{columns_to_use}' to create '{new_column}'.")
                if isinstance(df, dd.DataFrame):
                    df[new_column] = df[columns_to_use].map(func, meta=(new_column, object))
                else:
                    df[new_column] = df[columns_to_use].apply(func)
            elif isinstance(columns_to_use, list):
                # Multiple columns transformation
                logger.debug(f"Applying transformation on multiple columns {columns_to_use} to create '{new_column}'.")
                if isinstance(df, dd.DataFrame):
                    df[new_column] = df[columns_to_use].apply(func, axis=1, meta=(new_column, object))
                else:
                    df[new_column] = df[columns_to_use].apply(func, axis=1)
            else:
                logger.error(f"Invalid type for columns_to_use: {columns_to_use}. Expected str or list of str.")
                raise ValueError(f"Invalid type for columns_to_use: {columns_to_use}. Expected str or list of str.")

            logger.debug(f"Successfully applied transformation for '{new_column}'.")
        except Exception as e:
            logger.error(f"Error applying transformation for column '{new_column}': {e}")
            raise

    logger.info("All transformations applied successfully.")
    return df