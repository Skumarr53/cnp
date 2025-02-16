import os
from typing import Dict, Optional, List, Tuple, Callable, Any, Union
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    BooleanType,
    TimestampType,
    ArrayType,
    MapType,
    DataType,
)
#from loguru import logger

# Configure logging

def pd_udf_wrapper(func, schema, udf_type=PandasUDFType.SCALAR):
    @pandas_udf(schema, udf_type)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
  

def initialize_spark_session(app_name="Optimized_NLI_Inference", 
                     shuffle_partitions=200, gpu_amount=1, 
                     task_gpu_amount=0.8, executor_memory="4g", 
                     driver_memory="2g", executor_cores=1, 
                     memory_overhead="512m", dynamic_allocation="false"):
    """
    Initialize a Spark session with specified configurations.

    This function initializes a new Spark session using the provided configuration parameters. 
    If a Spark session with the specified `app_name` already exists, it returns the existing session.

    Args:
        app_name (str, optional): 
            The name of the Spark application. Defaults to "Optimized_NLI_Inference".
        shuffle_partitions (int, optional): 
            Number of partitions to use for shuffle operations. Defaults to 200.
        gpu_amount (float, optional): 
            Amount of GPU resources to allocate to each executor. Defaults to 1.
        task_gpu_amount (float, optional): 
            Amount of GPU resources to allocate to each task. Defaults to 0.8.
        executor_memory (str, optional): 
            Memory allocated to each executor (e.g., '4g'). Defaults to "4g".
        driver_memory (str, optional): 
            Memory allocated to the driver (e.g., '2g'). Defaults to "2g".
        executor_cores (int, optional): 
            Number of CPU cores allocated to each executor. Defaults to 1.
        memory_overhead (str, optional): 
            Amount of memory overhead to allocate per executor (e.g., '512m'). Defaults to "512m".
        dynamic_allocation (str, optional): 
            Enable dynamic allocation of executors ('true' or 'false'). Defaults to "false".

    Returns:
        SparkSession: 
            The initialized Spark session.

    Raises:
        Exception: 
            If the Spark session initialization fails.

    Example:
        >>> spark_session = initialize_spark_session(
        ...     app_name="MySparkApp",
        ...     shuffle_partitions=100,
        ...     executor_memory="8g"
        ... )
        >>> print(spark_session)
        <pyspark.sql.session.SparkSession object at 0x...>
    """
    try:
        spark = (SparkSession.builder.appName(app_name)
                    .config("spark.sql.shuffle.partitions", shuffle_partitions)
                    .config("spark.executor.resource.gpu.amount", gpu_amount)
                    .config("spark.task.resource.gpu.amount", task_gpu_amount)
                    .config("spark.executor.memory", executor_memory)
                    .config("spark.driver.memory", driver_memory)
                    .config("spark.executor.cores", executor_cores)
                    .config("spark.yarn.executor.memoryOverhead", memory_overhead)
                    .config("spark.dynamicAllocation.enabled", dynamic_allocation)
                    .getOrCreate())
        
        spark.sparkContext.setLogLevel("DEBUG")
        print("Spark session initialized successfully.")
        return spark
    
    except Exception as e:
        print(f"Failed to initialize Spark session: {e}")
        raise e

# Example usage
# spark_session = initialize_spark()  # To create a new session
# spark_session = initialize_spark(spark=spark_session)  # To reuse an existing session

def get_default_dtype_mapping() -> Dict[str, DataType]:
    """
    Retrieve the default mapping from Pandas dtypes to Spark DataTypes.

    This function provides a predefined dictionary that maps common Pandas data types to their 
    corresponding Spark DataTypes. This mapping is used to facilitate the conversion of 
    Pandas DataFrames to Spark DataFrames with appropriate schema definitions.

    Returns:
        Dict[str, DataType]: 
            A dictionary mapping Pandas dtype strings to Spark `DataType` objects.

    Example:
        >>> dtype_mapping = get_default_dtype_mapping()
        >>> print(dtype_mapping['int64'])
        LongType()
    """
    return {
        'object': StringType(),
        'int64': LongType(),
        'int32': IntegerType(),
        'float64': DoubleType(),
        'float32': FloatType(),
        'bool': BooleanType(),
        'datetime64[ns]': TimestampType(),
        'timedelta[ns]': StringType(),  # Spark does not have a timedelta type
        'string': StringType()
    }

def equivalent_type(
    column_name: str,
    pandas_dtype: str,
    column_mapping: Optional[Dict[str, str]] = None,
) -> DataType:
    """
    Determine the Spark DataType for a given column based on column name and Pandas dtype.

    This function maps a Pandas dtype to the corresponding Spark DataType. If a `column_mapping` 
    is provided, it prioritizes mapping based on the column name. If no specific mapping is found, 
    it falls back to the default dtype mapping. If still unresolved, it defaults to `StringType`.

    Args:
        column_name (str): 
            The name of the column.
        pandas_dtype (str): 
            The Pandas dtype of the column.
        column_mapping (Optional[Dict[str, str]], optional): 
            A dictionary mapping column names to Spark type identifier keywords. 
            This allows for custom type mappings based on column names. Defaults to None.

    Returns:
        DataType: 
            The corresponding Spark `DataType` for the column.

    Example:
        >>> dtype = equivalent_type("age", "int64")
        >>> print(dtype)
        LongType()
        
        >>> custom_mapping = {"price": "float"}
        >>> dtype = equivalent_type("price", "int64", column_mapping=custom_mapping)
        >>> print(dtype)
        FloatType()
    """


    # Fallback to default dtype mapping
    default_dtype_mapping = get_default_dtype_mapping()
    if pandas_dtype in default_dtype_mapping:
        print(f"Pandas dtype '{pandas_dtype}' for column '{column_name}' mapped to default Spark type '{default_dtype_mapping[pandas_dtype]}'.")
        return default_dtype_mapping[pandas_dtype]
    
    # Check if column name has a custom keyword mapping
    if column_mapping:
        if column_name in column_mapping:
            spark_type = column_mapping[column_name]
            if spark_type:
                print(f"Column '{column_name}'  mapped to '{spark_type}'.")
                return spark_type
        else:
            for key in column_mapping:
                if key.lower() in column_name.lower():
                    spark_type = column_mapping[key]
                    print(f"Column '{column_name}' mapped to '{spark_type}'.")
                    return spark_type
    
    # Fallback to StringType if no mapping is found
    print(f"No mapping found for column '{column_name}' with Pandas dtype '{pandas_dtype}'. Using StringType.")
    return StringType()

def define_structure(
    column_name: str,
    pandas_dtype: str,
    column_mapping: Optional[Dict[str, str]] = None,
) -> StructField:
    """
    Create a StructField for a Spark StructType schema based on column name and Pandas dtype.

    This function determines the appropriate Spark `DataType` for a given column by utilizing 
    the `equivalent_type` function. It then constructs a `StructField` with the column name, 
    determined `DataType`, and sets it as nullable.

    Args:
        column_name (str): 
            The name of the column.
        pandas_dtype (str): 
            The Pandas dtype of the column.
        column_mapping (Optional[Dict[str, str]], optional): 
            A dictionary mapping column names to Spark type identifier keywords for custom type mappings. 
            Defaults to None.

    Returns:
        StructField: 
            A Spark `StructField` object with the column name, determined `DataType`, and nullable set to True.

    Example:
        >>> field = define_structure("age", "int64")
        >>> print(field)
        StructField("age", LongType(), True)
        
        >>> custom_mapping = {"price": "float"}
        >>> field = define_structure("price", "int64", column_mapping=custom_mapping)
        >>> print(field)
        StructField("price", FloatType(), True)
    """
    spark_type = equivalent_type(column_name, pandas_dtype, column_mapping)
    return StructField(column_name, spark_type, nullable=True)

def pandas_to_spark(
    pandas_df: pd.DataFrame,
    column_type_mapping: Optional[Dict[str, str]] = None,
    spark: Optional[SparkSession] =  None
) -> DataFrame:
    """
    Convert a Pandas DataFrame to a Spark DataFrame with customizable type mappings.

    This function transforms a Pandas DataFrame into a Spark DataFrame by defining a Spark schema 
    based on the Pandas dtypes and any provided custom column type mappings. It ensures that each 
    column in the resulting Spark DataFrame has an appropriate `DataType`.

    Predefined Pandas dtype to Spark DataType mapping:
        - 'object'            -> StringType()
        - 'int64'             -> LongType()
        - 'int32'             -> IntegerType()
        - 'float64'           -> DoubleType()
        - 'float32'           -> FloatType()
        - 'bool'              -> BooleanType()
        - 'datetime64[ns]'    -> TimestampType()
        - 'timedelta[ns]'     -> StringType()  
        - 'string'            -> StringType()

    Custom keyword to Spark DataType mapping:
        - 'arr[str]'          -> ArrayType(StringType())
        - 'arr[int]'          -> ArrayType(IntegerType())
        - 'arr[long]'         -> ArrayType(LongType())
        - 'arr[float]'        -> ArrayType(FloatType())
        - 'arr[double]'       -> ArrayType(DoubleType())
        - 'arr[bool]'         -> ArrayType(BooleanType())
        - 'map[str,int]'      -> MapType(StringType(), IntegerType())
        - 'map[str,str]'      -> MapType(StringType(), StringType())

    Args:
        pandas_df (pd.DataFrame): 
            The Pandas DataFrame to convert.
        column_type_mapping (Optional[Dict[str, str]], optional): 
            Optional mapping from column names to type identifier keywords.
            Example: {'FILT_MD': 'arr_str', 'stats': 'map_str_int'}
        spark (Optional[SparkSession], optional): 
            The active SparkSession. If None, a new SparkSession will be created. Defaults to None.

    Returns:
        DataFrame: 
            The resulting Spark DataFrame with the defined schema.
    
    Raises:
        ValueError: 
            If there's an issue during the conversion process.
    """
    print("Starting conversion from Pandas to Spark DataFrame.")
    
    if spark is None:
        spark = (SparkSession.builder.appName('test').getOrCreate())

    columns = pandas_df.columns
    dtypes = pandas_df.dtypes

    struct_fields = []
    for column, dtype in zip(columns, dtypes):
        pandas_dtype = str(dtype)
        field = define_structure(column, pandas_dtype, column_type_mapping)
        struct_fields.append(field)
    
    schema = StructType(struct_fields)
    print(f"Constructed Spark schema: {schema}")
    
    try:
        spark_df = spark.createDataFrame(pandas_df, schema=schema)
        print("Successfully converted Pandas DataFrame to Spark DataFrame.")
        return spark_df
    except Exception as e:
        print(f"Error converting Pandas DataFrame to Spark DataFrame: {e}")
        raise ValueError(f"Conversion failed: {e}") from e


def convert_columns_to_timestamp(
    df: DataFrame,
    columns_formats: Dict[str, str],
    overwrite: bool = True
) -> DataFrame:
    """
    Convert specified columns in a Spark DataFrame to timestamp type using provided formats.

    This function applies the `to_timestamp` transformation to each column specified in the 
    `columns_formats` dictionary. It either overwrites the existing columns with the converted 
    timestamp values or creates new columns with a '_ts' suffix based on the `overwrite` parameter.

    Args:
        df (DataFrame): 
            The input Spark DataFrame.
        columns_formats (Dict[str, str]): 
            A dictionary where keys are column names to be converted, and values are the corresponding 
            timestamp formats.
            Example:
                {
                    "DATE": "yyyy-MM-dd",
                    "REPORT_DATE": "yyyy-MM-dd HH mm ss",
                    "EVENT_DATETIME_UTC": "yyyy-MM-dd HH mm ss"
                }
        overwrite (bool, optional): 
            Whether to overwrite the existing column with the transformed column. 
            If `False`, a new column with a '_ts' suffix will be created. Defaults to `True`.

    Returns:
        DataFrame: 
            The Spark DataFrame with specified columns converted to timestamp type.

    Raises:
        ValueError: 
            If `columns_formats` is empty.
        KeyError: 
            If a specified column does not exist in the DataFrame.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("TimestampConversion").getOrCreate()
        >>> data = [
        ...     ("2023-01-01", "2023-01-01 12 00 00", "2023-01-01 12 00 00"),
        ...     ("2023-02-01", "2023-02-01 13 30 45", "2023-02-01 13 30 45")
        ... ]
        >>> columns = ["DATE", "REPORT_DATE", "EVENT_DATETIME_UTC"]
        >>> df = spark.createDataFrame(data, schema=columns)
        >>> columns_to_convert = {
        ...     "DATE": "yyyy-MM-dd",
        ...     "REPORT_DATE": "yyyy-MM-dd HH mm ss",
        ...     "EVENT_DATETIME_UTC": "yyyy-MM-dd HH mm ss"
        ... }
        >>> converted_df = convert_columns_to_timestamp(df, columns_to_convert)
        >>> converted_df.printSchema()
        root
         |-- DATE: timestamp (nullable = true)
         |-- REPORT_DATE: timestamp (nullable = true)
         |-- EVENT_DATETIME_UTC: timestamp (nullable = true)
        >>> converted_df.show(truncate=False)
        +-------------------+-------------------+---------------------+
        |DATE               |REPORT_DATE        |EVENT_DATETIME_UTC   |
        +-------------------+-------------------+---------------------+
        |2023-01-01 00:00:00|2023-01-01 12:00:00|2023-01-01 12:00:00  |
        |2023-02-01 00:00:00|2023-02-01 13:30:45|2023-02-01 13:30:45  |
        +-------------------+-------------------+---------------------+
    """
    if not columns_formats:
        print("No columns and formats provided for timestamp conversion.")
        raise ValueError("The 'columns_formats' dictionary cannot be empty.")

    for column, fmt in columns_formats.items():
        if column not in df.columns:
            print(f"Column '{column}' does not exist in the DataFrame.")
            raise KeyError(f"Column '{column}' not found in the DataFrame.")

        if overwrite:
            print(f"Converting column '{column}' to timestamp with format '{fmt}'. Overwriting existing column.")
            df = df.withColumn(column, F.to_timestamp(F.col(column), fmt))
        else:
            new_column = f"{column}_ts"
            print(f"Converting column '{column}' to timestamp with format '{fmt}'. Creating new column '{new_column}'.")
            df = df.withColumn(new_column, F.to_timestamp(F.col(column), fmt))

    print("Timestamp conversion completed successfully.")
    return df




def sparkdf_apply_transformations(
    spark_df: DataFrame,
    transformations: List[Tuple[str, Union[str, List[str]], Callable[..., Any]]],
    error_on_missing: bool = True
) -> DataFrame:
    """
    Apply a series of transformations to a PySpark DataFrame based on provided specifications.

    Each transformation is defined by a tuple containing:
        - The name of the new or existing column to be created or overwritten.
        - The name(s) of the column(s) to be used as input for the transformation.
        - The transformation function to apply.

    Args:
        spark_df (DataFrame): The input PySpark DataFrame to transform.
        transformations (List[Tuple[str, Union[str, List[str]], Callable[..., Any]]]):
            A list of transformation specifications. Each specification is a tuple:
                (new_column_name, input_columns, transformation_function)
                
                - 'new_column_name' (str): The name of the column to create or overwrite.
                - 'input_columns' (str or List[str]): The column name(s) to pass as arguments to the transformation function.
                - 'transformation_function' (Callable): A function that takes one or more 'Column' objects and returns a 'Column'.
        error_on_missing (bool, optional): 
            If 'True', the function will raise a 'KeyError' if any of the specified input columns are missing in the DataFrame.
            If 'False', missing columns will be skipped with a warning.
            Defaults to 'True'.

    Returns:
        DataFrame: The transformed PySpark DataFrame with all specified transformations applied.

    Raises:
        KeyError: If 'error_on_missing' is 'True' and any specified input columns are missing.
        TypeError: If 'transformations' is not a list of tuples with the required structure.

    Example:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StringType

        # Initialize SparkSession
        spark = SparkSession.builder.appName("TransformationExample").getOrCreate()

        # Sample DataFrame
        data = [
            ("Hello World", "2023-01-01"),
            ("PySpark Transformations", "2023-02-01")
        ]
        columns = ["text_column", "date_column"]
        df = spark.createDataFrame(data, schema=columns)

        # Define UDFs
        def to_upper(text):
            return text.upper() if text else text

        def extract_year(date_str):
            return date_str.split("-")[0] if date_str else None

        to_upper_udf = udf(to_upper, StringType())
        extract_year_udf = udf(extract_year, StringType())

        # Define transformations
        transformations = [
            ("text_upper", "text_column", to_upper_udf),
            ("year_extracted", "date_column", extract_year_udf),
            ("combined", ["text_column", "date_column"], lambda text, date: F.concat(text, F.lit(" - "), date))
        ]

        # Apply transformations
        transformed_df = apply_transformations(df, transformations)

        # Show results
        transformed_df.show(truncate=False)

        **Output:**
        +-------------------------+-----------+-----------+-----------------------------+
        |text_column              |date_column|text_upper |year_extracted |combined              |
        +-------------------------+-----------+-----------+-----------------------------+
        |Hello World              |2023-01-01 |HELLO WORLD|2023           |Hello World - 2023-01-01|
        |PySpark Transformations  |2023-02-01 |PYSPARK TRANSFORMATIONS|2023           |PySpark Transformations - 2023-02-01|
        +-------------------------+-----------+-----------+-----------------------------+

    """
    if not isinstance(transformations, list):
        print("Transformations should be provided as a list of tuples.")
        raise TypeError("Transformations should be a list of tuples.")

    for idx, transformation in enumerate(transformations):
        if not (isinstance(transformation, tuple) and len(transformation) == 3):
            print(
                f"Each transformation should be a tuple of (new_column_name, input_columns, transformation_function). "
                f"Error at transformation index {idx}: {transformation}"
            )
            raise TypeError(
                "Each transformation must be a tuple of (new_column_name, input_columns, transformation_function)."
            )

        new_column, input_columns, transformation_func = transformation

        # Normalize input_columns to a list
        if isinstance(input_columns, str):
            input_columns = [input_columns]
        elif isinstance(input_columns, list):
            if not all(isinstance(col_name, str) for col_name in input_columns):
                print(f"All input column names must be strings. Error at transformation index {idx}: {transformation}")
                raise TypeError("All input column names must be strings.")
        else:
            print(
                f"input_columns must be either a string or a list of strings. Error at transformation index {idx}: {transformation}"
            )
            raise TypeError("input_columns must be either a string or a list of strings.")

        # Check for missing columns
        missing_cols = [col_name for col_name in input_columns if col_name not in spark_df.columns]
        if missing_cols:
            message = f"Missing input columns {missing_cols} for transformation '{new_column}'."
            if error_on_missing:
                print(message)
                raise KeyError(message)
            else:
                print(message)
                continue  # Skip this transformation

        # Prepare Column objects
        column_objs = [F.col(col_name) for col_name in input_columns]

        # Apply the transformation function
        try:
            print(f"Applying transformation for column '{new_column}' using columns {input_columns}.")
            transformed_column = transformation_func(*column_objs)
            spark_df = spark_df.withColumn(new_column, transformed_column)
            print(f"Transformation for column '{new_column}' applied successfully.")
        except Exception as e:
            print(f"Error applying transformation for column '{new_column}': {e}")
            raise e

    print("All transformations have been applied successfully.")
    return spark_df


def check_spark_dataframe_for_records(spark_df: DataFrame,
                                      datetime_col: str = 'PARSED_DATETIME_EASTERN_TZ') -> None:
    """
    Check if the provided Spark DataFrame contains records and log relevant information.

    This function verifies the presence of records in the Spark DataFrame. If records are present, it logs the minimum and maximum dates based on the specified datetime column, along with the row and column counts. If no records are found, it logs a warning and exits the notebook or process.

    Args:
        spark_df (DataFrame): 
            The Spark DataFrame to check.
        datetime_col (str, optional): 
            The name of the datetime column to use for calculating date ranges. 
            Defaults to 'PARSED_DATETIME_EASTERN_TZ'.

    Raises:
        ValueError: 
            If the input is not a valid Spark DataFrame.
        KeyError: 
            If the specified `datetime_col` does not exist in the DataFrame.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("DataFrameCheckExample").getOrCreate()
        >>> data = [
        ...     ("2023-01-01 08:00:00"),
        ...     ("2023-02-01 09:30:00")
        ... ]
        >>> columns = ["PARSED_DATETIME_EASTERN_TZ"]
        >>> df = spark.createDataFrame(data, schema=columns)
        >>> check_spark_dataframe_for_records(df)
        The data spans from 2023-01-01 08:00:00 to 2023-02-01 09:30:00 and has 2 rows and 1 columns.
    """
    if not isinstance(spark_df, DataFrame):
        raise ValueError("The provided input is not a valid Spark DataFrame.")

    if spark_df.head(1):  # Check if the DataFrame is not empty
        # Calculate min and max parsed datetime
        min_parsed_date = spark_df.agg({datetime_col: "min"}).collect()[0][0]
        max_parsed_date = spark_df.agg({datetime_col: "max"}).collect()[0][0]
        row_count = spark_df.count()
        col_count = len(spark_df.columns)

        # Log the information
        print(f'The data spans from {min_parsed_date} to {max_parsed_date} '
                    f'and has {row_count} rows and {col_count} columns.')
    else:
        from databricks.sdk.runtime import dbutils
        print('No new transcripts to parse.')
        dbutils.notebook.exit(1)  # Exit the notebook with a non-zero status
        os._exit(1)  # Terminate the process 