# centralized_nlp_package/common_utils/date_utils.py
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Tuple
from loguru import logger

def get_date_range(years_back: int = 0, months_back: int = 0) -> Tuple[str, str]:
    """
    Calculate a date range based on the current date minus specified years and/or months.

    This function computes the start and end dates by subtracting the given number of years and months from the current date. The dates are returned in 'YYYY-MM-DD' format, representing the first day of the respective months.

    Args:
        years_back (int, optional): 
            Number of years to go back from the current date. Defaults to 0.
        months_back (int, optional): 
            Number of months to go back from the current date. Defaults to 0.

    Returns:
        Tuple[str, str]: 
            A tuple containing:
            - `start_date` (str): The start date in 'YYYY-MM-DD' format.
            - `end_date` (str): The end date in 'YYYY-MM-DD' format.

    Example:
        >>> from centralized_nlp_package.common_utils.date_utils import get_date_range
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
    Format a datetime object into a string with 'YYYY-MM-DD' format.

    This function takes a `datetime` object and returns its string representation in the 'YYYY-MM-DD' format.

    Args:
        date (datetime): 
            The datetime object to format.

    Returns:
        str: 
            The formatted date string in 'YYYY-MM-DD' format.

    Example:
        >>> from centralized_nlp_package.common_utils.date_utils import format_date
        >>> from datetime import datetime
        >>> formatted = format_date(datetime(2023, 9, 15))
        >>> print(formatted)
        '2023-09-15'
    """
    formatted_date = date.strftime('%Y-%m-%d')
    logger.debug(f"Formatted date: {formatted_date}")
    return formatted_date

def get_current_date_str() -> str:
    """
    Retrieve the current date as a string in 'YYYYMMDD' format.

    This function returns the current date formatted as a string without any separators, following the 'YYYYMMDD' pattern.

    Returns:
        str: 
            The current date in 'YYYYMMDD' format.

    Example:
        >>> from centralized_nlp_package.common_utils.date_utils import get_current_date_str
        >>> current_date = get_current_date_str()
        >>> print(current_date)
        '20250127'
    """
    return datetime.today().strftime('%Y%m%d')