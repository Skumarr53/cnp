import os
import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
#from loguru import logger
from centralized_nlp_package.utils import (
    load_file,
    get_date_range,
    format_date,
    format_string_template,
    query_constructor,
    df_remove_rows_with_keywords,
    df_apply_transformations,
)


# Test for load_file function
def test_load_file(tmp_path):
    # Create a temporary SQL file
    sql_file = tmp_path / "test_query.sql"
    sql_file.write_text("SELECT * FROM test_table;")

    # Test loading the file
    result = load_file(str(sql_file))
    assert result == "SELECT * FROM test_table;"

    # Test FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_file(str(tmp_path / "non_existent.sql"))


# Test for get_date_range function
def test_get_date_range():
    min_date, max_date = get_date_range(years_back=1, months_back=0)
    expected_year = datetime.now().year - 1
    assert min_date.startswith(f"{expected_year}-")
    assert max_date == datetime.now().strftime("%Y-%m-01")

    min_date, max_date = get_date_range(months_back=6)
    assert min_date.startswith(f"{datetime.now().year}-")
    assert max_date == datetime.now().strftime("%Y-%m-01")

    # Test default values
    min_date, max_date = get_date_range()
    assert min_date.startswith(f"{datetime.now().year}-")
    assert max_date == datetime.now().strftime("%Y-%m-01")


# Test for format_date function
def test_format_date():
    date = datetime(2023, 1, 15)
    formatted = format_date(date)
    assert formatted == "2023-01-15"


# Test for format_string_template function
def test_format_string_template():
    template = "path/to/{year}/{month}/{day}/file.txt"
    result = format_string_template(template, year=2023, month=5, day=15)
    assert result == "path/to/2023/5/15/file.txt"

    # Test with missing placeholder
    with pytest.raises(ValueError):
        format_string_template(template, year=2023, month=5)


# Test for query_constructor function
def test_query_constructor(tmp_path):
    # Test using a query string
    query = "SELECT * FROM users WHERE name = '{name}'"
    result = query_constructor(query, name="Alice")
    assert result == "SELECT * FROM users WHERE name = 'Alice'"

    # Test using a file path
    sql_file = tmp_path / "test_query.sql"
    sql_file.write_text("SELECT * FROM users WHERE age = {age}")
    result = query_constructor(str(sql_file), age=25)
    assert result == "SELECT * FROM users WHERE age = 25"

    # Test FileNotFoundError
    with pytest.raises(FileNotFoundError):
        query_constructor(str(tmp_path / "non_existent.sql"))


# Test for df_remove_rows_with_keywords function
def test_df_remove_rows_with_keywords():
    data = {
        "text": ["This is a test", "Another example", "Keyword present", "No keyword here"]
    }
    df = pd.DataFrame(data)
    keywords = ["Keyword"]

    result_df = df_remove_rows_with_keywords(df, "text", keywords)

    # Check that the row with "Keyword present" is removed
    assert len(result_df) == 3
    assert "Keyword present" not in result_df["text"].values

    # Test with non-existent column
    with pytest.raises(ValueError):
        df_remove_rows_with_keywords(df, "non_existent_column", keywords)

    # Test with keywords that are not present in the column
    keywords_not_present = ["NotInColumn"]
    with pytest.warns(UserWarning):
        result_df = df_remove_rows_with_keywords(df, "text", keywords_not_present)
    assert len(result_df) == 4  # No rows should be removed as the keyword is not present


# Test for df_apply_transformations function
def test_df_apply_transformations():
    data = {
        "value": [1, 2, 3, 4],
        "multiplier": [10, 20, 30, 40]
    }
    df = pd.DataFrame(data)

    transformations = {
        "new_column": lambda row: row["value"] * row["multiplier"],
        "value": lambda x: x + 1
    }

    result_df = df_apply_transformations(df, transformations)

    # Check that transformations were applied correctly
    assert "new_column" in result_df.columns
    assert result_df["new_column"].tolist() == [10, 40, 90, 160]
    assert result_df["value"].tolist() == [2, 3, 4, 5]

    # Test with invalid transformation
    transformations = {"invalid_column": "not_a_function"}
    with pytest.raises(Exception):
        df_apply_transformations(df, transformations)


if __name__ == "__main__":
    pytest.main()
