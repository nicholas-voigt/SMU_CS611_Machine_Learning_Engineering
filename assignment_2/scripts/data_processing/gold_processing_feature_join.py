import os
import argparse

from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F

from data_loading import load_data
from helpers_data_processing import build_partition_name, validate_date, pyspark_df_info
from data_configuration import silver_data_dirs, gold_data_dirs


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
    print(f"\nLoading all feature data for date: {date} from silver directory...")
    table_dfs = {}
    for table_name, table_path in silver_data_dirs.items():
        partition_name = build_partition_name('silver', table_name, args.date, 'parquet')
        table_dfs[table_name] = load_data(spark=spark, input_directory=table_path, partition=partition_name)

    # Step 1: For customers with label, select only snapshot dates which correspond to loan application date (months on book = 0)
    print("\nExtracting customers with labels & their loan start date from label store...")
    relevant_dates = table_dfs['loan_data'] \
        .filter(F.col('mob') == 0) \
        .select("Customer_ID", "snapshot_date") \
        .withColumnRenamed("snapshot_date", "loan_start_date")
    print(f"Number of customers where loan start date is given: {relevant_dates.count()}")

    # Filter & aggregate clickstream data to include only relevant dates
    print("Filtering clickstream data for customers with label & aggregating clickstream data to loan start date...")
    clickstream_data_aggr = table_dfs['clickstream_data'] \
        .join(relevant_dates, on="Customer_ID", how="inner") \
        .filter(F.col('snapshot_date') <= F.col('loan_start_date')) \
        .groupBy("Customer_ID", "snapshot_date") \
        .agg(
            *[F.avg(F.col(f"fe_{i}")).alias(f"fe_{i}_avg") for i in range(1, 21)]
        )
    
    # Join all dataframes to Feature Store
    print("\nJoining current partitions to Feature Store...")
    df = table_dfs['customer_attributes'].join(table_dfs['customer_financials'], on=(["Customer_ID", "snapshot_date"]), how="inner")
    df_feature_store = df.join(clickstream_data_aggr, on=(["Customer_ID", "snapshot_date"]), how="left_outer")
    print("Joined Customer Attributes, Customer Financials & Clickstream Data to Feature Store")
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