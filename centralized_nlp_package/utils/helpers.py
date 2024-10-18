# centralized_nlp_package/utils/helpers.py
import os
import re
import ast
import dask.dataframe as dd
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple, Union, List, Callable
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
        with open(file_path, 'r') as file:return file.read()
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
        ValueError: If the specified column is not found in the DataFrame.
        Warning: If any keyword is not found in the specified column.
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
    transformations: List[Tuple[str, Callable, bool]],
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Applies a set of transformations to a DataFrame based on the given list of transformation tuples.

    Args:
        df (Union[pd.DataFrame, dd.DataFrame]): The DataFrame to apply transformations on.
        transformations (List[Tuple[str, Callable, bool]]): 
            A list of tuples where each tuple contains:
                - column (str): The column name to transform.
                - func (Callable): The transformation function.
                - use_row (bool): If True, apply the function row-wise (axis=1). 
                                   If False, apply column-wise (axis=0).
        dask_processing (bool, optional): Flag indicating whether to process using Dask. Defaults to False.

    Returns:
        Union[pd.DataFrame, dd.DataFrame]: The DataFrame with applied transformations.

    Raises:
        Exception: Re-raises any exception that occurs during the transformation process after logging.
    Notes:
    - Ensure that the `transformations` list does not contain tuples with non-callable functions.
    - When applying multiple transformations to the same column, transformations are applied in the order they appear in the list.
    - For Dask DataFrames, ensure that transformations are compatible with Dask's lazy evaluation model.
    """

    for transformation in transformations:
        if len(transformation) == 3:
            column, func, use_row = transformation
        elif len(transformation) == 2:
            column, func = transformation
            use_row = False  # Default to column-wise
        else:
            logger.error(f"Invalid transformation tuple: {transformation}. Expected 2 or 3 elements.")
            continue  # Skip invalid transformation tuples

        if not callable(func):
            logger.error(f"Error: Transformation function for column '{column}' is not callable. Skipping.")
            continue 

        try:
            # Apply row-wise if lambda function mentions multiple columns, otherwise apply column-wise
            if isinstance(df, dd.DataFrame):
                # Apply the transformation function on the entire row (axis=1) if needed
                df[column] = (df.apply(func, meta = (column, object), axis=1) 
                              if use_row 
                              else df[column].apply(func, meta = (column, object)))
            elif isinstance(df, pd.DataFrame):
                df[column] = (df.apply(func, axis=1) 
                            if use_row 
                            else df[column].apply(func))
            else:
                logger.error(f"Error: Transformation for column '{column}' is not callable. Skipping.")
        except Exception as e:
            # Handle any errors during transformation
            logger.error(f"Error occurred while transforming column '{column}': {e}")
            raise
    return  df



def df_apply_transformations(
    df: Union[pd.DataFrame, dd.DataFrame],
    transformations: List[Tuple[str, Union[str, List[str]], Callable]],
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Applies a set of transformations to a DataFrame based on the given list of transformation tuples.

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
        Exception: Re-raises any exception that occurs during the transformation process after logging.

    Notes:
        - Ensure that the `transformations` list contains tuples with callable functions.
        - When applying multiple transformations to the same column, transformations are applied in the order they appear in the list.
        - For Dask DataFrames, ensure that transformations are compatible with Dask's lazy evaluation model.
    """

    for transformation in transformations:
        if len(transformation) == 3:
            new_column, columns_to_use, func = transformation
        else:
            logger.error(f"Invalid transformation tuple: {transformation}. Expected 3 elements.")
            continue  # Skip invalid transformation tuples

        if not callable(func):
            logger.error(f"Error: Transformation function for new column '{new_column}' is not callable. Skipping.")
            continue 

        try:
            if isinstance(columns_to_use, str):
                # Single column transformation
                if isinstance(df, dd.DataFrame):
                    df[new_column] = df[columns_to_use].apply(func, meta=(new_column, object))
                else:
                    df[new_column] = df[columns_to_use].apply(func)
            elif isinstance(columns_to_use, list):
                # Multiple columns transformation
                if isinstance(df, dd.DataFrame):
                    df[new_column] = df[columns_to_use].apply(func, axis=1, meta=(new_column, object))
                else:
                    df[new_column] = df[columns_to_use].apply(func, axis=1)
            else:
                logger.error(f"Invalid columns_to_use: {columns_to_use}. Expected str or list of str.")
                continue
        except Exception as e:
            # Handle any errors during transformation
            logger.error(f"Error occurred while transforming to new column '{new_column}': {e}")
            raise
    return df
