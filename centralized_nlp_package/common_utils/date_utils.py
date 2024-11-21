# centralized_nlp_package/common_utils/date_utils.py
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Tuple
from loguru import logger

def get_date_range(years_back: int = 0, months_back: int = 0) -> Tuple[str, str]:
    """
    Calculates the date range based on the current date minus specified years and/or months.

    Args:
        years_back (int, optional): Number of years to go back from the current date. Defaults to 0.
        months_back (int, optional): Number of months to go back from the current date. Defaults to 0.

    Returns:
        Tuple[str, str]: A tuple containing the start date and end date in 'YYYY-MM-DD' format.

    Example:
        >>> from centralized_nlp_package.common_utils import get_date_range
        >>> start_date, end_date = get_date_range(years_back=1, months_back=2) ## exmaple current date is 2023-10-01
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
        >>> from centralized_nlp_package.common_utils import format_date
        >>> from datetime import datetime
        >>> format_date(datetime(2023, 9, 15))
        '2023-09-15'
    """
    formatted_date = date.strftime('%Y-%m-%d')
    logger.debug(f"Formatted date: {formatted_date}")
    return formatted_date

def get_current_date_str() -> str:
    """
    Returns the current date in 'YYYYMMDD' format.

    Returns:
        str: The current date in 'YYYYMMDD' format.
    """
    return datetime.today().strftime('%Y%m%d')