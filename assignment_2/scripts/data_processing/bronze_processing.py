# This function processes the source data and saves it to the bronze layer.
# It is used in the main.py file.

import os
import argparse
from datetime import datetime
import pyspark
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
    output_directory = '../../datamart/bronze/' + bronze_table_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # load data
    print('Loading data for date:', date)
    df = spark.read.csv(input_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(date, 'row count:', df.count())

    # save bronze table to datamart
    partition_name = 'bronze_' + bronze_table_name + '_' + date.replace('-','_') + '.csv'
    filepath = output_directory + '/' + partition_name
    df.toPandas().to_csv(filepath, index=False)

    print('Bronze table created for date:', date, 'and saved to:', filepath)
    return


if __name__ == "__main__":

    # get input arguments
    parser = argparse.ArgumentParser(description='Add data to bronze table.')
    parser.add_argument('--date', type=str, required=True, help='The date for which to add data, in the format YYYY-MM-DD')
    parser.add_argument('--input_path', type=str, required=True, help='The path to the input file. File should be in CSV format.')
    parser.add_argument('--bronze_name', type=str, required=True, help='The name of the destination table. This should match the name of the subdirectory in the bronze directory.')
    args = parser.parse_args()

    # validate input arguments
    if not args.date or not args.input_path or not args.bronze_name:
        raise ValueError("All arguments --date, --input_path, and --bronze_name are required.")
    
    # create spark session
    spark = SparkSession.builder \
        .appName("Bronze Processing") \
        .getOrCreate()

    # set log level
    spark.sparkContext.setLogLevel("ERROR")

    # call function to add data to bronze table
    add_data_bronze(date=args.date, input_file_path=args.input_path, bronze_table_name=args.bronze_name, spark=spark)

    # Stop the Spark session
    spark.stop()
    print(f"\n\n---Bronze Store completed successfully for {args.date}---\n\n")
