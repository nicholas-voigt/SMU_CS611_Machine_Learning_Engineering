# The following functions are used to process the data in the silver layer. They are called in the main.py file.

import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, StringType, DateType
from utils.data_loading import load_data


def add_clickstream_data_silver(spark: SparkSession, date: str, bronze_directory: str, silver_directory: str):
    '''
    Function to process clickstream data from bronze table to silver table. Enforces schema and data quality checks.
    Args:
        spark (SparkSession): Spark session object.
        date (str): Date for which data is being processed (corresponds to partition). Format: 'YYYY-MM-DD'.
        bronze_directory (str): Path to the bronze directory.
        silver_directory (str): Path to the silver directory.
    '''

    # Check input arguments
    if not os.path.exists(bronze_directory):
        raise FileNotFoundError(f"Bronze directory {bronze_directory} does not exist.")
    
    # Load data from bronze directory
    partition = 'bronze_clickstream_data_' + date.replace("-", "_") + '.csv'
    df = load_data(spark, bronze_directory, partition)
    bronze_count = df.count()
    if df is None or bronze_count == 0:
        raise ValueError(f"No data found in bronze directory for partition {partition}.")
    
    # Data quality checks

    # enforce schema
    column_type_map = {
        "fe_1": IntegerType(),
        "fe_2": IntegerType(),
        "fe_3": IntegerType(),
        "fe_4": IntegerType(),
        "fe_5": IntegerType(),
        "fe_6": IntegerType(),
        "fe_7": IntegerType(),
        "fe_8": IntegerType(),
        "fe_9": IntegerType(),
        "fe_10": IntegerType(),
        "fe_11": IntegerType(),
        "fe_12": IntegerType(),
        "fe_13": IntegerType(),
        "fe_14": IntegerType(),
        "fe_15": IntegerType(),
        "fe_16": IntegerType(),
        "fe_17": IntegerType(),
        "fe_18": IntegerType(),
        "fe_19": IntegerType(),
        "fe_20": IntegerType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType()
        }
    for column, dtype in column_type_map.items():
        df = df.withColumn(column, col(column).cast(dtype))

    # Handle null values (0 for numerical, remove if customer_id or snapshot_date are corrupted)
    for column in column_type_map.keys():
        if column == "Customer_ID" or column == "snapshot_date":
            df = df.filter(col(column).isNotNull())
        else:
            df = df.fillna(0, subset=[column])
    
    # Remove duplicates
    df = df.dropDuplicates(["Customer_ID", "snapshot_date"])
    
    # check for row count after cleaning
    silver_count = df.count()
    if silver_count != bronze_count:
        print(f"Warning: Row count changed from {bronze_count} to {silver_count} after cleaning for partition {partition}.")

    # Ensure that silver directory exists 
    os.makedirs(os.path.dirname(silver_directory), exist_ok=True)

    # Save the cleaned data to the silver directory as parquet
    silver_partition_name = 'silver_clickstream_data_' + date.replace("-", "_") + '.parquet'
    silver_filepath = os.path.join(silver_directory, silver_partition_name)

    df.write.mode("overwrite").parquet(silver_filepath)
    print(f"Successfully processed and saved data to {silver_filepath}.")

    return
