import os
import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from helpers_data_processing import validate_date, build_partition_name, pyspark_df_info
from data_configuration import source_data_files, bronze_data_dirs


def add_data_bronze(date: str, type: str, spark: SparkSession):
    """
    Function to add data to a bronze table according to medallion architecture.
    Args:
        date (str): The date for which to add data, in the format "YYYY-MM-DD".
        type (str): The data type to be processed.
        spark (SparkSession): The Spark session object.
    Returns nothing.
    """

    # prepare arguments
    snapshot_date = validate_date(date, output_DateType=True)
    input_file = source_data_files[type]
    output_directory = bronze_data_dirs[type]
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # load data
    print('Loading data for date:', date)
    df = spark.read.csv(input_file, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(f"Loaded data from {input_file}. Row count: {df.count()}")

    # Show DataFrame information
    pyspark_df_info(df)

    # save bronze table to datamart
    partition_name = build_partition_name('bronze', type, date, 'csv')
    filepath = os.path.join(output_directory, partition_name)
    df.toPandas().to_csv(filepath, index=False)

    print(f"Bronze table created for {type} on date {date} and saved to {filepath}")
    return


if __name__ == "__main__":

    # get input arguments
    parser = argparse.ArgumentParser(description='Add data to bronze table.')
    parser.add_argument('--date', type=str, required=True, help='The date for which to add data, in the format YYYY-MM-DD')
    parser.add_argument('--type', type=str, required=True, choices=['clickstream_data', 'customer_attributes', 'customer_financials', 'loan_data'],
                        help='Type of data to process: clickstream_data, customer_attributes, customer_financials, or loan_data.')
    args = parser.parse_args()

    # validate input arguments
    if not args.date or not args.type:
        raise ValueError("All arguments --date and --type are required.")

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Bronze Processing") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # call function to add data to bronze table
    add_data_bronze(date=args.date, type=args.type, spark=spark)

    # Stop the Spark session
    spark.stop()
    print(f"\n\n---Bronze Store completed successfully---\n\n")
