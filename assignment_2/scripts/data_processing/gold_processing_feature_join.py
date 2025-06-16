import os
import argparse

from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F

from data_loading import load_data
from helpers_data_processing import build_partition_name, validate_date, pyspark_df_info
from configurations import silver_data_dirs, gold_data_dirs


if __name__ == "__main__":
    # get input arguments
    parser = argparse.ArgumentParser(description='Join silver feature tables to gold feature store.')
    parser.add_argument('--date', type=str, required=True, help='The date for which to join data, in the format YYYY-MM-DD')
    args = parser.parse_args()

    # validate input arguments
    if not args.date:
        raise ValueError("Argument --date is required.")
    date = validate_date(args.date)

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Gold Feature Joining") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # load all data
    print(f"\nLoading all feature data for date: {date} from silver directory...\n")
    table_dfs = {}
    for table_name, table_path in silver_data_dirs.items():
        if table_name != 'loan_data':
            partition_name = build_partition_name('silver', table_name, args.date, 'parquet')
            table_dfs[table_name] = load_data(spark=spark, input_directory=table_path, partition=partition_name)

    # Join all dataframes to Feature Store
    print("\nJoining current partitions to Feature Store...\n")

    # Step 1: Prepare Attributes Table by renaming columns
    df_attributes_renamed = table_dfs['customer_attributes'] \
        .withColumnRenamed("snapshot_date", "attr_effective_date") \
        .withColumnRenamed("Customer_ID", "attr_Customer_ID")

    # Step 2: Join Clickstream with Attributes (As-Of Join)
    window_attr = Window.partitionBy(F.col("cs.Customer_ID"), F.col("cs.snapshot_date")).orderBy(F.col("attr.attr_effective_date").desc())

    df_cs_attr = table_dfs['clickstream_data'].alias("cs") \
        .join(
            df_attributes_renamed.alias("attr"), # Alias df_attributes_renamed as "attr" for this join
            (F.col("cs.Customer_ID") == F.col("attr.attr_Customer_ID")) & # Use "attr." prefix
            (F.col("cs.snapshot_date") >= F.col("attr.attr_effective_date")), # Use "attr." prefix
            "left_outer"
        ) \
        .withColumn("attr_rank", F.row_number().over(window_attr)) \
        .filter(F.col("attr_rank") == 1) \
        .drop("attr_rank", "attr_Customer_ID", "attr_effective_date")

    print("Joined Clickstream with Attributes via As-Of Join")
    pyspark_df_info(df_cs_attr)

    # Step 3: Prepare Aliases for Financials Table
    df_financials_renamed = table_dfs['customer_financials'] \
        .withColumnRenamed("snapshot_date", "fin_effective_date") \
        .withColumnRenamed("Customer_ID", "fin_Customer_ID")

    # Step 4: Join Clickstream-Attributes with Financials (As-Of Join)
    window_fin = Window.partitionBy(F.col("cs_attr.Customer_ID"), F.col("cs_attr.snapshot_date")).orderBy(F.col("fin.fin_effective_date").desc())

    df_feature_store = df_cs_attr.alias("cs_attr") \
        .join(
            df_financials_renamed.alias("fin"), # Alias df_financials_renamed as "fin" for this join
            (F.col("cs_attr.Customer_ID") == F.col("fin.fin_Customer_ID")) &
            (F.col("cs_attr.snapshot_date") >= F.col("fin.fin_effective_date")),
            "left_outer"
        ) \
        .withColumn("fin_rank", F.row_number().over(window_fin)) \
        .filter(F.col("fin_rank") == 1) \
        .drop("fin_rank", "fin_Customer_ID", "fin_effective_date")

    print("Joined Clickstream-Attributes with Financials via As-Of Join")
    pyspark_df_info(df_feature_store)

    # Ensure that gold directory exists
    os.makedirs(os.path.dirname(gold_data_dirs['feature_store']), exist_ok=True)

    # Save the cleaned data to the gold directory as parquet
    partition_name = build_partition_name('gold', 'feature_store', args.date, 'parquet')
    gold_filepath = os.path.join(gold_data_dirs['feature_store'], partition_name)

    df_feature_store.write.mode("overwrite").parquet(gold_filepath)
    print(f"Gold feature store saved to {gold_filepath}.")

    # Stop the Spark session
    spark.stop()
    print(f"\n\n---Gold feature store completed successfully for {args.date}---\n\n")