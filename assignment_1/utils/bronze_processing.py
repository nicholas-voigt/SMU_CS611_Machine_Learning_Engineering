# This function processes the source data and saves it to the bronze layer.
# It is used in the main.py file.

import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def add_data_bronze(date: str, input_file_path: str, bronze_table_name: str, spark: SparkSession):
    '''
    Function to add data to a bronze table according to medallion architecture.
    Args:
        date (str): The date for which to add data, in the format "YYYY-MM-DD".
        input_file_path (str): The path to the input file. File should be in CSV format.
        bronze_table_name (str): The name of the destination table. This should match the name of the subdirectory in the bronze directory.
        spark (SparkSession): The Spark session object.
    Returns nothing.
    '''

    # prepare arguments
    ## date formatting
    snapshot_date = datetime.strptime(date, "%Y-%m-%d")
    if not isinstance(snapshot_date, datetime):
        raise ValueError("Date must be in the format YYYY-MM-DD")
    ## check if file paths exists
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file {input_file_path} does not exist.")
    output_directory = 'datamart/bronze/' + bronze_table_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # load data
    df = spark.read.csv(input_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(date + 'row count:', df.count())

    # save bronze table to datamart
    partition_name = 'bronze_' + bronze_table_name + '_' + date.replace('-','_') + '.csv'
    filepath = output_directory + '/' + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return