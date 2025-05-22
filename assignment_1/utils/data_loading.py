# This function loads data from a CSV file into a Spark DataFrame

import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import col


def load_data(spark: SparkSession, input_directory: str, partition: str | None) -> DataFrame:
    """
    Load data from a CSV file into a Spark DataFrame.
    Args:
        spark (SparkSession): The Spark session object.
        input_directory (str): The directory to the input files. Files should be in CSV format.
        partition (str): A specific partition to load. If not provided, all data in the given directory will be loaded.
    Returns:
        DataFrame: A Spark DataFrame containing the loaded data.
    """

    # Check if the input directory exists
    if not os.path.exists(input_directory):
        raise FileNotFoundError(f"Input directory {input_directory} does not exist.")

    # If partition is provided, check if file exists and load it
    if partition:
        file_path = os.path.join(input_directory, partition)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File for partition {partition} does not exist in {input_directory}.")
        # Load the data from the CSV file
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        print(f"Loaded data from {file_path}. Row count: {df.count()}")
        return df
    
    # If no partition is provided, load all data in the directory
    else:
        all_files_path = os.path.join(input_directory, "*.csv")
        print(f"Loading csv files with wildcard: {all_files_path}")
        df = spark.read.csv(all_files_path, header=True, inferSchema=True)
        
        print(f"Loaded data from all csv files in {input_directory}. Row count: {df.count()}")
        return df
