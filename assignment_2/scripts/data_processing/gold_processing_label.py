import os
import argparse 

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType

from data_loading import load_data


def add_gold_label_store(spark: SparkSession, date: str, silver_directory: str, gold_directory: str, dpd=30, mob=6):
    '''
    Function to process loan data from silver table to the gold label store.
    Args:
        spark (SparkSession): Spark session object.
        date (str): Date for which data is being processed (corresponds to partition). Format: 'YYYY-MM-DD'.
        silver_directory (str): Path to the silver directory.
        gold_directory (str): Path to the gold label store directory.
        dpd (int): Days past due for label creation. Default is 30.
        mob (int): Months on book for label creation. Default is 6.
    '''

    # Check input arguments
    if not os.path.exists(silver_directory):
        raise FileNotFoundError(f"Silver directory {silver_directory} does not exist.")
    
    # Load data from silver directory
    partition = 'silver_label_store_' + date.replace("-", "_") + '.parquet'
    df = load_data(spark, silver_directory, partition)
    silver_count = df.count()
    if df is None or silver_count == 0:
        raise ValueError(f"No data found in silver directory for partition {partition}.")

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # Ensure that gold label store directory exists 
    os.makedirs(os.path.dirname(gold_directory), exist_ok=True)

    # Save the cleaned data to the gold label store as parquet
    partition_name = "gold_label_store_" + date.replace('-','_') + '.parquet'
    gold_label_store_filepath = os.path.join(gold_directory, partition_name)

    df.write.mode("overwrite").parquet(gold_label_store_filepath)
    print(f"Successfully processed and saved data to {gold_label_store_filepath}.")

    return


# main function to run the script
if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="Process loan data for gold label store.")
    parser.add_argument("--date", type=str, required=True, help="Date for which data is being processed (YYYY-MM-DD).")
    parser.add_argument("--silver_directory", type=str, required=True, help="Path to the silver directory.")
    parser.add_argument("--gold_directory", type=str, required=True, help="Path to the gold label store directory.")
    args = parser.parse_args()

    # Create Spark session
    spark = SparkSession.builder \
        .appName("Gold Label Store Processing") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Call the function to process loan data
    add_gold_label_store(spark, args.date, args.silver_directory, args.gold_directory)

    # Stop the Spark session
    spark.stop()
    print(f"\n\n---Gold Label Store completed successfully for {args.date}---\n\n")