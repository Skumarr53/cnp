# centralized_nlp_package/utils/dataframe_utils.py
from typing import Any, Callable, List, Tuple, Union
import pandas as pd
import dask.dataframe as dd
from loguru import logger

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
        >>> import centralized_nlp_package.data_processing import df_remove_rows_with_keywords
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
        >>> from centralized_nlp_package.data_processing import df_apply_transformations
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


def concatenate_and_reset_index(dataframes: List[pd.DataFrame], drop_column: str = 'index') -> pd.DataFrame:
    """
    Concatenates multiple Pandas DataFrames, resets the index, and drops the old index column.

    This function performs the following steps:
        1. Concatenates the provided DataFrames along the rows.
        2. Resets the index of the concatenated DataFrame.
        3. Drops the specified column (default is 'index') resulting from the reset.

    Args:
        dataframes (List[pd.DataFrame]): A list of Pandas DataFrames to concatenate.
        drop_column (str, optional): The name of the column to drop after resetting the index.
                                      Defaults to 'index'.

    Returns:
        pd.DataFrame: The concatenated DataFrame with the index reset and the old index column dropped.

    Raises:
        ValueError: If the input list `dataframes` is empty.
        KeyError: If the `drop_column` does not exist in the concatenated DataFrame.

    Example:
        >>> rdf = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> cdf = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        >>> sdf = pd.DataFrame({'A': [9, 10], 'B': [11, 12]})
        >>> concatdf = concatenate_and_reset_index([rdf, cdf, sdf])
        >>> print(concatdf)
            A   B
        0   1   3
        1   2   4
        2   5   7
        3   6   8
        4   9  11
        5  10  12
    """
    if not dataframes:
        logger.error("No DataFrames provided for concatenation.")
        raise ValueError("The list of DataFrames to concatenate is empty.")

    # Concatenate the DataFrames along the rows (default axis=0)
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    logger.debug(f"Concatenated DataFrame shape: {concatenated_df.shape}")

    # Reset the index (though ignore_index=True already does this)
    concatenated_df.reset_index(inplace=True)
    logger.debug("Index has been reset.")

    # Attempt to drop the specified column
    try:
        concatenated_df.drop(columns=[drop_column], inplace=True)
        logger.debug(f"Dropped column '{drop_column}'.")
    except KeyError:
        logger.error(f"Column '{drop_column}' does not exist in the DataFrame.")
        raise KeyError(f"Column '{drop_column}' not found in the DataFrame.")

    return concatenated_df
