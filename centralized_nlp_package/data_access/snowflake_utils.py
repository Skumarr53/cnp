# centralized_nlp_package/data_access/snowflake_utils.py
import os
import re
from dotenv import load_dotenv
from centralized_nlp_package import config
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import pandas as pd
from snowflake.connector import connect
from typing import Any
from pathlib import Path
from loguru import logger
from centralized_nlp_package import config
from cryptography.fernet import Fernet
# from pyspark.sql import SparkSession


def initialize_spark():
    """
    Initializes a Spark session and configures Snowflake integration if not already initialized.

    This function sets up the global Spark session, Spark context, and Snowflake utilities.
    It also configures the default timezone to UTC.

    Example:
        initialize_spark_session()
    """
    global spark, sc, sfUtils

    # Check if Spark session is already initialized
    if spark is None:
        # Create Spark session
        spark = SparkSession.builder \
            .appName("SnowflakeIntegration") \
            .getOrCreate()

        # Get the Spark context
        sc = spark.sparkContext

        # Set Snowflake pushdown session
        sfUtils = sc._jvm.net.snowflake.spark.snowflake.Utils
        sc._jvm.net.snowflake.spark.snowflake.SnowflakeConnectorUtils.enablePushdownSession(spark)

        # Set the default timezone to UTC
        zone = sc._jvm.java.util.TimeZone
        zone.setDefault(sc._jvm.java.util.TimeZone.getTimeZone("UTC"))
        logger.info("Spark session initialized.")
    else:
        logger.info("Spark session already initialized; reusing existing session.")

def get_private_key() -> str:
    """
    Retrieves and processes the Snowflake private key from AKV.
    
    Returns:
        str: The private key in a format suitable for Snowflake authentication.
    """
    # Retrieve encrypted private key and password from AKV
    key_file = dbutils.secrets.get(scope="id-secretscope-dbk-pr4707-prod-work", key=config.snowflake_key)
    pwd = dbutils.secrets.get(scope="id-secretscope-dbk-pr4707-prod-work", key=config.snowflake_pwd)
    
    # Load the private key using the retrieved password
    p_key = serialization.load_pem_private_key(
        key_file.encode('ascii'),
        password=pwd.encode(),
        backend=default_backend()
    )
    
    # Serialize the private key to PEM format without encryption
    pkb = p_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    # Decode and clean the private key string
    pkb = pkb.decode("UTF-8")
    pkb = re.sub("-*(BEGIN|END) PRIVATE KEY-*\n", "", pkb).replace("\n", "")
    
    return pkb


def get_snowflake_options():
    """
    Returns a dictionary of Snowflake options.
    """
    private_key = get_private_key()

    _config = config.lib_config.development.snowflake
    snowflake_options = {
        'sfURL': f'{_config.account}.snowflakecomputing.com',
        'sfUser': _config.user,
        "pem_private_key": private_key,
        'sfDatabase': _config.database,
        'sfSchema': _config.schema,
        "sfTimezone": "spark",
        'sfRole': _config.role  # Optional if needed
    }
    return  snowflake_options

def read_from_snowflake_spark(query: str, spark):
    # TODO: read_from_snowflake_spark(query: str, spark: SparkSession) -> 'pyspark.sql.DataFrame':
    """
    Executes a SQL query on Snowflake and returns the result as a Spark DataFrame.

    Args:
        query (str): The SQL query to execute.
        config (Config): Configuration object containing Snowflake credentials.
        spark (SparkSession): The active Spark session.

    Returns:
        pyspark.sql.DataFrame: Query result.
    """
    logger.info("Reading data from Snowflake using Spark.")

    snowflake_options = get_snowflake_options()

    try:
        logger.debug(f"Executing query: {query}")
        df_spark = spark.read.format("snowflake") \
            .options(**snowflake_options) \
            .option("query", query) \
            .load()
        logger.info("Query executed successfully and Spark DataFrame created.")
    except Exception as e:
        logger.error(f"Error executing query on Snowflake: {e}")
        raise

    return df_spark

def write_to_snowflake_spark(df, table_name: str, mode: str = 'append') -> None:
    # TODO: write_to_snowflake_spark(df: 'pyspark.sql.DataFrame', table_name: str, mode: str = 'append') -> None:
    """
    Writes a Spark DataFrame to a Snowflake table.

    Args:
        df (pyspark.sql.DataFrame): The Spark DataFrame to write to Snowflake.
        table_name (str): The target table name in Snowflake.
        spark (SparkSession): The active Spark session.
        mode (str): Specifies the behavior if the table already exists:
                    'append', 'overwrite', 'error', or 'ignore'. Default is 'append'.

    Returns:
        None
    """
    logger.info(f"Writing Spark DataFrame to Snowflake table: {table_name}.")
    snowflake_options = get_snowflake_options()

    try:
        df.write.format("snowflake") \
            .options(**snowflake_options) \
            .option("dbtable", table_name) \
            .mode(mode) \
            .save()
        logger.info(f"DataFrame written successfully to {table_name}.")
    except Exception as e:
        logger.error(f"Error writing Spark DataFrame to Snowflake: {e}")
        raise

def truncate_or_merge_table( query):
    snowflake_options = get_snowflake_options()
    df=sfUtils.runQuery(snowflake_options, query)
    result="Truncate or Merge Complete"
    return result