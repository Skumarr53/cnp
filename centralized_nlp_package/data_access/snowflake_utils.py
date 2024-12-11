# centralized_nlp_package/data_access/snowflake_utils.py

import os
import re
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import pandas as pd
from snowflake.connector import connect
from loguru import logger
from pyspark.sql import SparkSession, DataFrame
from functools import wraps
from databricks.sdk.runtime import *

from centralized_nlp_package import config
from centralized_nlp_package.utils import determine_environment

# Module-level variables to hold the Spark session and Snowflake utilities
_spark_session: Optional[SparkSession] = None
_sfUtils: Optional[Any] = None  # Replace 'Any' with the appropriate type if available
    

def singleton(cls):
    """
    A decorator to make a class a Singleton by ensuring only one instance exists.
    """
    instances = {}

    @wraps(cls)
    def get_instance(*args, **kwargs):
        global _spark_session, _sfUtils
        if cls not in instances:
            #logger.info(f"Creating a new instance of {cls.__name__}")
            instances[cls] = cls(*args, **kwargs)
            _spark_session = instances[cls].spark
            _sfUtils = instances[cls].sfUtils
        # else:
            #logger.info(f"Using existing instance of {cls.__name__}")
        return instances[cls]

    return get_instance


@singleton
class SparkSessionManager:
    """
    Singleton class to manage a single Spark session across the application.
    """

    def __init__(self, app_name: str = "SnowflakeIntegration") -> None:
        """
        Initializes the Spark session and configures Snowflake integration.

        Args:
            app_name (str): The name of the Spark application.
        """
        self.spark: SparkSession = SparkSession.builder \
            .appName(app_name) \
            .getOrCreate()

        self.setup_spark()

    def setup_spark(self) -> None:
        """
        Configures Spark session settings and Snowflake integration.
        """
        global _sfUtils
        #logger.info("Configuring Spark session settings...")

        sc = self.spark.sparkContext

        # Configure Snowflake pushdown session
        self.sfUtils = sc._jvm.net.snowflake.spark.snowflake.Utils
        # sc._jvm.net.snowflake.spark.snowflake.SnowflakeConnectorUtils.enablePushdownSession(self.spark)
        sc._jvm.net.snowflake.spark.snowflake.SnowflakeConnectorUtils.enablePushdownSession(sc._jvm.org.apache.spark.sql.SparkSession.builder().getOrCreate())

        # Set the default timezone to UTC
        zone = sc._jvm.java.util.TimeZone
        zone.setDefault(sc._jvm.java.util.TimeZone.getTimeZone("UTC"))

        #logger.info("Spark session configured and Snowflake integration enabled.")


def with_spark_session(func):
    """
    Decorator to ensure that the Spark session is initialized before function execution.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _spark_session
        if _spark_session is None:
            #logger.info("Spark session not initialized. Initializing now...")
            SparkSessionManager(app_name="SnowflakeIntegration")
        else:
            print("Spark session already initialized; reusing existing session.")
        return func(*args, **kwargs)
    return wrapper

@with_spark_session
def retrieve_snowflake_private_key(config) -> str:
    """
    Retrieves and processes the Snowflake private key from Azure Key Vault (AKV).

    This function fetches the encrypted private key and password from AKV, decrypts the key,
    and formats it for Snowflake authentication.

    Returns:
        str: The private key in PEM format suitable for Snowflake authentication.

    Example:
        >>> private_key = retrieve_snowflake_private_key()
    """
    global _spark_session  # Access Spark session if needed

    # Retrieve encrypted private key and password from AKV
    try:
        key_file = dbutils.secrets.get(scope="id-secretscope-dbk-pr4707-prod-work", key=config.key)
        pwd = dbutils.secrets.get(scope="id-secretscope-dbk-pr4707-prod-work", key=config.password)
        logger.debug("Retrieved secrets from AKV successfully.")
    except Exception as e:
        logger.error(f"Error retrieving secrets from AKV: {e}")
        raise

    # Load the private key using the retrieved password
    try:
        p_key = serialization.load_pem_private_key(
            key_file.encode('ascii'),
            password=pwd.encode(),
            backend=default_backend()
        )
        logger.debug("Private key loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading private key: {e}")
        raise

    # Serialize the private key to PEM format without encryption
    try:
        pkb = p_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        logger.debug("Private key serialized to PEM format.")
    except Exception as e:
        logger.error(f"Error serializing private key: {e}")
        raise

    # Decode and clean the private key string
    pkb = pkb.decode("UTF-8")
    pkb = re.sub("-*(BEGIN|END) PRIVATE KEY-*\n", "", pkb).replace("\n", "")
    logger.debug("Private key decoded and cleaned.")

    return pkb


@with_spark_session
def get_snowflake_connection_options(database: str = 'EDS_PROD' , schema: str = 'QUANT') -> Dict[str, str]:
    """
    Constructs and returns a dictionary of Snowflake connection options.

    This includes the Snowflake account URL, user credentials, private key, database, schema,
    timezone, and role.

    Returns:
        Dict[str, str]: A dictionary containing Snowflake connection parameters.

    Example:
        >>> snowflake_options = get_snowflake_connection_options()
    """
    global _spark_session  # Access Spark session if needed

    determined_env = determine_environment()
    _config = config.lib_config.snowflake[determined_env]
    private_key = retrieve_snowflake_private_key(_config)

    
    snowflake_options = {
        'sfURL': f'{_config.account}.snowflakecomputing.com',
        'sfUser': _config.user,
        "pem_private_key": private_key,
        'sfDatabase': database,
        'sfSchema': schema,
        "sfTimezone": "spark",
        'sfRole': _config.role  # Optional if needed
    }
    logger.debug("Snowflake connection options constructed.")

    return snowflake_options


@with_spark_session
def read_from_snowflake(query: str) -> DataFrame:
    """
    Executes a SQL query on a specified Snowflake database and schema, returning the result as a Spark DataFrame.
    
    This function utilizes the provided Spark session to execute the given SQL query against the specified 
    database and schema in Snowflake. It ensures that the query is executed within the correct context 
    by setting the appropriate connection options.
    
    Args:
        query (str): The SQL query to execute.
        database (str): The target database name in Snowflake where the query will be executed.
        schema (str): The target schema name within the specified database in Snowflake.
    
    Returns:
        DataFrame: The result of the SQL query as a Spark DataFrame.
    
    Raises:
        Exception: If there is an error executing the query on Snowflake.
    
    Example:
        >>> from centralized_nlp_package.data_access import read_from_snowflake
        >>> df = read_from_snowflake(
        ...     query="SELECT * FROM analytics_db.public.sales_data",
        ...     database="analytics_db",
        ...     schema="public"
        ... )
        >>> df.show()
    """
    global _spark_session  # Access Spark session

    logger.info("Reading data from Snowflake using Spark.")

    snowflake_options = get_snowflake_connection_options()

    try:
        logger.debug(f"Executing query: {query}")
        df_spark = _spark_session.read.format("net.snowflake.spark.snowflake") \
            .options(**snowflake_options) \
            .option("query", query) \
            .load()
        logger.info("Query executed successfully and Spark DataFrame created.")
    except Exception as e:
        logger.error(f"Error executing query on Snowflake: {e}")
        raise

    return df_spark

@with_spark_session
def write_dataframe_to_snowflake(df: DataFrame, database: str, schema: str, table_name: str,  mode: str = 'append') -> None:
    """
    Writes a Spark DataFrame to a specified Snowflake table.

    This function utilizes the provided Spark session to write the given DataFrame to a 
    Snowflake table. It allows specifying the target database and schema, ensuring that 
    the data is correctly directed to the intended location within Snowflake.

    Args:
        df (DataFrame): The Spark DataFrame to write to Snowflake.
        database (str): The target database name in Snowflake.
        schema (str): The target schema name within the specified database in Snowflake.
        table_name (str): The target table name in Snowflake where the DataFrame will be written.
        mode (str, optional): Specifies the behavior if the table already exists.
                              Options are:
                                - 'append': Append the data to the existing table.
                                - 'overwrite': Overwrite the existing table with the new data.
                                - 'error': Throw an error if the table exists.
                                - 'ignore': Ignore the operation if the table exists.
                              Default is 'append'.

    Returns:
        None

    Raises:
        Exception: If there is an error writing the DataFrame to Snowflake.

    Example:
        >>> from centralized_nlp_package.data_access import write_dataframe_to_snowflake
        >>> write_dataframe_to_snowflake(
        ...     df, 
        ...     database="analytics_db", 
        ...     schema="public", 
        ...     table_name="sales_data", 
        ...     mode="append"
        ... )
    """
    global _spark_session  # Access Spark session

    logger.info(f"Writing Spark DataFrame to Snowflake table: {table_name}.")
    snowflake_options = get_snowflake_connection_options(database, schema)

    try:
        df.write.format("net.snowflake.spark.snowflake") \
            .options(**snowflake_options) \
            .option("dbtable", table_name) \
            .mode(mode) \
            .save()
        logger.info(f"DataFrame written successfully to {table_name}.")
    except Exception as e:
        logger.error(f"Error writing Spark DataFrame to Snowflake: {e}")
        raise


@with_spark_session
def execute_truncate_or_merge_query(query: str, database: str, schema: str) -> str:
    """
    Executes a TRUNCATE or MERGE SQL query on a specified Snowflake database and schema.

    This function runs the provided SQL query (either TRUNCATE or MERGE) against the 
    specified database and schema in Snowflake using the established Spark session. 
    It ensures that the query is executed within the correct context.

    Args:
        query (str): The SQL query to execute. Typically, this will be a TRUNCATE or MERGE statement.
        database (str): The target database name in Snowflake where the query will be executed.
        schema (str): The target schema name within the specified database in Snowflake.

    Returns:
        str: A confirmation message indicating the successful completion of the operation.

    Raises:
        Exception: If there is an error executing the TRUNCATE or MERGE query on Snowflake.

    Example:
        >>> from centralized_nlp_package.data_access import execute_truncate_or_merge_query
        >>> execute_truncate_or_merge_query(
        ...     query="TRUNCATE TABLE analytics_db.public.sales_data",
        ...     database="analytics_db",
        ...     schema="public"
        ... )
    """
    global _spark_session  # Access Spark session

    snowflake_options = get_snowflake_connection_options(database, schema)

    try:
        logger.debug(f"Executing TRUNCATE or MERGE query: {query}")
        _spark_session._jvm.net.snowflake.spark.snowflake.Utils.runQuery(snowflake_options, query)
        logger.info("Truncate or Merge operation completed successfully.")
    except Exception as e:
        logger.error(f"Error executing TRUNCATE or MERGE query on Snowflake: {e}")
        raise