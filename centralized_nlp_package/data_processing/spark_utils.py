from typing import Dict, Optional, List, Tuple, Callable, Any, Union
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
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
from loguru import logger

# Define the keyword to Spark DataType mapping
KEYWORD_TO_SPARK_TYPE: Dict[str, DataType] = {
    'arr[str]': ArrayType(StringType()),
    'arr[int]': ArrayType(IntegerType()),
    'arr[long]': ArrayType(LongType()),
    'arr[float]': ArrayType(FloatType()),
    'arr[double]': ArrayType(DoubleType()),
    'arr[bool]': ArrayType(BooleanType()),
    'map[str,int]': MapType(StringType(), IntegerType()),
    'map[str,str]': MapType(StringType(), StringType()),
    'double': DoubleType(),
    'float64': DoubleType()
    # Add more mappings as needed
}

# Configure logging

def initialize_spark_session(app_name="Optimized_NLI_Inference", 
                     shuffle_partitions="200", gpu_amount="1", 
                     task_gpu_amount="0.8", executor_memory="4g", 
                     driver_memory="2g", executor_cores="1", 
                     memory_overhead="512m", dynamic_allocation="false"):
    """
    Initializes a Spark session with specified configurations.
    
    Args:
        spark (SparkSession, optional): An existing Spark session to use. 
                                         If None, a new session will be created.
        app_name (str): The name of the Spark application.
        shuffle_partitions (int, optional): Number of partitions to use for shuffle operations.
        gpu_amount (float, optional): Amount of GPU resources to allocate to executors.
        task_gpu_amount (float, optional): Amount of GPU resources to allocate to tasks.
        executor_memory (str, optional): Memory allocated to each executor (e.g., '4g').
        driver_memory (str, optional): Memory allocated to the driver (e.g., '2g').
        executor_cores (int, optional): Number of cores allocated to each executor.
        memory_overhead (str, optional): Amount of memory overhead to allocate per executor (e.g., '512m').
        dynamic_allocation (bool, optional): Enable dynamic allocation of executors (default is False).
    
    Returns:
        SparkSession: The initialized or existing Spark session.
    
    Raises:
        Exception: If the Spark session initialization fails.
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
        logger.info("Spark session initialized successfully.")
        return spark
    
    except Exception as e:
        logger.error(f"Failed to initialize Spark session: {e}")
        raise e

# Example usage
# spark_session = initialize_spark()  # To create a new session
# spark_session = initialize_spark(spark=spark_session)  # To reuse an existing session

def get_default_dtype_mapping() -> Dict[str, DataType]:
    """
    Returns the default mapping from Pandas dtypes to Spark DataTypes.
    
    Returns:
        Dict[str, DataType]: Mapping of Pandas dtypes to Spark DataTypes.
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
    }

def keyword_to_datatype(keyword: str) -> Optional[DataType]:
    """
    Converts a keyword to the corresponding Spark DataType.
    
    Args:
        keyword (str): The type identifier keyword.
    
    Returns:
        Optional[DataType]: The corresponding Spark DataType, or None if keyword is invalid.
    """
    dtype = KEYWORD_TO_SPARK_TYPE.get(keyword.lower())
    if dtype:
        logger.debug(f"Keyword '{keyword}' mapped to Spark DataType '{dtype}'.")
    else:
        logger.warning(f"Keyword '{keyword}' is not recognized. It will be ignored.")
    return dtype

def equivalent_type(
    column_name: str,
    pandas_dtype: str,
    column_mapping: Optional[Dict[str, str]] = None,
) -> DataType:
    """
    Determines the Spark DataType for a given column based on column name and Pandas dtype.
    Priority is given to column name mapping over dtype mapping.
    
    Args:
        column_name (str): Name of the column.
        pandas_dtype (str): Pandas dtype of the column.
        column_mapping (Optional[Dict[str, str]]): Mapping from column names to type identifier keywords.
    
    Returns:
        DataType: Corresponding Spark DataType.
    """
    # Check if column name has a custom keyword mapping
    if column_mapping:
        if column_name in column_mapping:
                keyword = column_mapping[column_name]
                spark_type = keyword_to_datatype(keyword)
                if spark_type:
                    logger.debug(f"Column '{column_name}' uses custom keyword '{keyword}' mapped to '{spark_type}'.")
                    return spark_type
                else:
                    logger.warning(f"Column '{column_name}' has an invalid keyword '{keyword}'. Falling back to default mapping.")
        else:
            key = [key in column_name for key in column_mapping]
            if key:
                keyword = column_mapping[key[0]] if isinstance(key, list) else column_mapping[key]
                spark_type = keyword_to_datatype(keyword)
                if spark_type:
                    logger.debug(f"Column '{column_name}' uses custom keyword '{keyword}' mapped to '{spark_type}'.")
                    return spark_type
                else:
                    logger.warning(f"Column '{column_name}' has an invalid keyword '{keyword}'. Falling back to default mapping.")

    # Fallback to default dtype mapping
    default_dtype_mapping = get_default_dtype_mapping()
    if pandas_dtype in default_dtype_mapping:
        logger.debug(f"Pandas dtype '{pandas_dtype}' for column '{column_name}' mapped to default Spark type '{default_dtype_mapping[pandas_dtype]}'.")
        return default_dtype_mapping[pandas_dtype]
    
    # Fallback to StringType if no mapping is found
    logger.warning(f"No mapping found for column '{column_name}' with Pandas dtype '{pandas_dtype}'. Using StringType.")
    return StringType()

def define_structure(
    column_name: str,
    pandas_dtype: str,
    column_mapping: Optional[Dict[str, str]] = None,
) -> StructField:
    """
    Creates a StructField for a Spark StructType schema.
    
    Args:
        column_name (str): Name of the column.
        pandas_dtype (str): Pandas dtype of the column.
        column_mapping (Optional[Dict[str, str]]): Mapping from column names to type identifier keywords.
    
    Returns:
        StructField: StructField with column name and determined Spark DataType.
    """
    spark_type = equivalent_type(column_name, pandas_dtype, column_mapping)
    return StructField(column_name, spark_type, nullable=True)

def pandas_to_spark(
    pandas_df: pd.DataFrame,
    spark: SparkSession,
    column_type_mapping: Optional[Dict[str, str]] = None,
) -> DataFrame:
    """
    Converts a Pandas DataFrame to a Spark DataFrame with customizable type mappings.

    If a column name is present in the 'column_type_mapping', its Spark DataType will be determined
    based on the provided type identifier keyword. If a column name is not present in the mapping,
    its type will be determined based on the Pandas dtype using a predefined dtype mapping.

    Predefined Pandas dtype to Spark DataType mapping:
        - 'object'            -> StringType()
        - 'int64'             -> LongType()
        - 'int32'             -> IntegerType()
        - 'float64'           -> DoubleType()
        - 'float32'           -> FloatType()
        - 'bool'              -> BooleanType()
        - 'datetime64[ns]'    -> TimestampType()
        - 'timedelta[ns]'     -> StringType()  

    Args:
        pandas_df (pd.DataFrame): The Pandas DataFrame to convert.
        spark (SparkSession): The active SparkSession.
        column_type_mapping (Optional[Dict[str, str]]): Optional mapping from column names to type identifier keywords.
            Example: {'FILT_MD': 'arr_str', 'stats': 'map_str_int'}
            
            custom keyword to Spark DataType mapping:

            - 'arr[str]'     -> ArrayType(StringType())
            - 'arr[int]'     -> ArrayType(IntegerType())
            - 'arr[long]'    -> ArrayType(LongType())
            - 'arr[float]'   -> ArrayType(FloatType())
            - 'arr[double]'  -> ArrayType(DoubleType())
            - 'arr[bool]'    -> ArrayType(BooleanType())
            - 'map[str,int]' -> MapType(StringType(), IntegerType())
            - 'map[str,str]' -> MapType(StringType(), StringType())
            

    Returns:
        DataFrame: The resulting Spark DataFrame.

    Raises:
        ValueError: If there's an issue during the conversion process.
    """
    logger.info("Starting conversion from Pandas to Spark DataFrame.")
    
    columns = pandas_df.columns
    dtypes = pandas_df.dtypes

    struct_fields = []
    for column, dtype in zip(columns, dtypes):
        pandas_dtype = str(dtype)
        field = define_structure(column, pandas_dtype, column_type_mapping)
        struct_fields.append(field)
    
    schema = StructType(struct_fields)
    logger.debug(f"Constructed Spark schema: {schema}")
    
    try:
        spark_df = spark.createDataFrame(pandas_df, schema=schema)
        logger.info("Successfully converted Pandas DataFrame to Spark DataFrame.")
        return spark_df
    except Exception as e:
        logger.error(f"Error converting Pandas DataFrame to Spark DataFrame: {e}")
        raise ValueError(f"Conversion failed: {e}") from e


def convert_columns_to_timestamp(
    df: DataFrame,
    columns_formats: Dict[str, str],
    overwrite: bool = True
) -> DataFrame:
    """
    Converts specified columns in a Spark DataFrame to timestamp type using provided formats.

    This function iterates over the provided dictionary of column names and their corresponding
    timestamp formats, applying the 'to_timestamp' transformation to each specified column.

    Args:
        df (DataFrame): The input Spark DataFrame.
        columns_formats (Dict[str, str]): A dictionary where keys are column names to be converted,
                                          and values are the corresponding timestamp formats.
                                          Example:
                                              {
                                                  "DATE": "yyyy-MM-dd",
                                                  "REPORT_DATE": "yyyy-MM-dd HH mm ss",
                                                  "EVENT_DATETIME_UTC": "yyyy-MM-dd HH mm ss"
                                              }
        overwrite (bool, optional): Whether to overwrite the existing column with the transformed column.
                                    If 'False', a new column with a suffix (e.g., '_ts') will be created.
                                    Defaults to 'True'.

    Returns:
        DataFrame: The Spark DataFrame with specified columns converted to timestamp type.

    Raises:
        ValueError: If 'columns_formats' is empty.
        KeyError: If a specified column does not exist in the DataFrame.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("ExampleApp").getOrCreate()
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
        logger.error("No columns and formats provided for timestamp conversion.")
        raise ValueError("The 'columns_formats' dictionary cannot be empty.")

    for column, fmt in columns_formats.items():
        if column not in df.columns:
            logger.error(f"Column '{column}' does not exist in the DataFrame.")
            raise KeyError(f"Column '{column}' not found in the DataFrame.")

        if overwrite:
            logger.info(f"Converting column '{column}' to timestamp with format '{fmt}'. Overwriting existing column.")
            df = df.withColumn(column, F.to_timestamp(F.col(column), fmt))
        else:
            new_column = f"{column}_ts"
            logger.info(f"Converting column '{column}' to timestamp with format '{fmt}'. Creating new column '{new_column}'.")
            df = df.withColumn(new_column, F.to_timestamp(F.col(column), fmt))

    logger.info("Timestamp conversion completed successfully.")
    return df




def sparkdf_apply_transformations(
    spark_df: DataFrame,
    transformations: List[Tuple[str, Union[str, List[str]], Callable[..., Any]]],
    error_on_missing: bool = True
) -> DataFrame:
    """
    Applies a series of transformations to a PySpark DataFrame based on the provided specifications.

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
        logger.error("Transformations should be provided as a list of tuples.")
        raise TypeError("Transformations should be a list of tuples.")

    for idx, transformation in enumerate(transformations):
        if not (isinstance(transformation, tuple) and len(transformation) == 3):
            logger.error(
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
                logger.error(f"All input column names must be strings. Error at transformation index {idx}: {transformation}")
                raise TypeError("All input column names must be strings.")
        else:
            logger.error(
                f"input_columns must be either a string or a list of strings. Error at transformation index {idx}: {transformation}"
            )
            raise TypeError("input_columns must be either a string or a list of strings.")

        # Check for missing columns
        missing_cols = [col_name for col_name in input_columns if col_name not in spark_df.columns]
        if missing_cols:
            message = f"Missing input columns {missing_cols} for transformation '{new_column}'."
            if error_on_missing:
                logger.error(message)
                raise KeyError(message)
            else:
                logger.warning(message)
                continue  # Skip this transformation

        # Prepare Column objects
        column_objs = [F.col(col_name) for col_name in input_columns]

        # Apply the transformation function
        try:
            logger.info(f"Applying transformation for column '{new_column}' using columns {input_columns}.")
            transformed_column = transformation_func(*column_objs)
            spark_df = spark_df.withColumn(new_column, transformed_column)
            logger.debug(f"Transformation for column '{new_column}' applied successfully.")
        except Exception as e:
            logger.error(f"Error applying transformation for column '{new_column}': {e}")
            raise e

    logger.info("All transformations have been applied successfully.")
    return spark_df


def create_spark_udf(function, return_type_key: str = 'arr[str]'):
    """
    Creates a Spark User Defined Function (UDF) from a given Python function.

    Args:
        function (callable): The Python function to be converted into a UDF.
        return_type_key (str): The return type of the UDF, specified as a key.
                                Default is 'arr[str]' for an array of strings.

    Returns:
        pyspark.sql.functions.UserDefinedFunction: The created Spark UDF.

    Raises:
        ValueError: If the return_type_key is not valid.
        Exception: If the UDF creation fails for any other reason.
    """
    # Validate the return_type_key
    if return_type_key.lower() not in KEYWORD_TO_SPARK_TYPE:
        logger.error(f"Invalid return type key: '{return_type_key}'. Valid keys are: {list(KEYWORD_TO_SPARK_TYPE.keys())}")
        raise ValueError(f"Invalid return type key: '{return_type_key}'. Valid keys are: {list(KEYWORD_TO_SPARK_TYPE.keys())}")

    try:
        spark_udf = F.udf(function, KEYWORD_TO_SPARK_TYPE[return_type_key.lower()])
        logger.info(f"Successfully created Spark UDF with return type: {return_type_key}")
        return spark_udf
    except Exception as e:
        logger.error(f"Failed to create Spark UDF: {e}")
        raise e


def check_spark_dataframe_for_records(spark_df: DataFrame,
                                      datetime_col: str = 'PARSED_DATETIME_EASTERN_TZ') -> None:
    """
    Checks if the provided Spark DataFrame contains records.
    If records are present, logs the minimum and maximum parsed date, 
    the row count, and the column count. If no records are found, 
    logs a warning and exits the notebook.

    Args:
        spark_df (DataFrame): The Spark DataFrame to check.

    Raises:
        ValueError: If the input is not a valid Spark DataFrame.
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
        logger.info(f'The data spans from {min_parsed_date} to {max_parsed_date} '
                    f'and has {row_count} rows and {col_count} columns.')
    else:
        logger.warning('No new transcripts to parse.')
        dbutils.notebook.exit(1)  # Exit the notebook with a non-zero status
        os._exit(1)  # Terminate the process 