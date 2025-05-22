# This file runs the entire data preprocessing pipeline.

import os
import glob
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame, Window

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_loading
import utils.bronze_processing
import utils.silver_processing
import utils.gold_processing


# Initialize SparkSession
print("Initializing Spark session & generating snapshot dates...")
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

print("\n\nProcessing Source Data to Bronze Tables...\n\n")

# Data Source:
source_tables = {
    'clickstream_data': 'data/feature_clickstream.csv',
    'customer_attributes': 'data/features_attributes.csv',
    'customer_financials': 'data/features_financials.csv',
    'loan_data': 'data/lms_loan_daily.csv'
}

# run bronze datalake backfill for each source table
for table, path in source_tables.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file {path} does not exist.")
    
    for date_str in dates_str_lst:
        utils.bronze_processing.add_data_bronze(date=date_str, input_file_path=path, bronze_table_name=table, spark=spark)


print("\n\n" + "-" * 50)
print("All Bronze tables loaded successfully.")
print("-" * 50)

# run silver datalake backfill
print("\n\nProcessing Data to Silver tables...\n\n")

# load clickstream data
print("\nLoading clickstream data...")
for date_str in dates_str_lst:
    utils.silver_processing.add_clickstream_data_silver(spark=spark, date=date_str, bronze_directory='datamart/bronze/clickstream_data/', silver_directory='datamart/silver/clickstream_data/')

# load customer attributes data
print("\nLoading customer attributes data...")
for date_str in dates_str_lst:
    utils.silver_processing.add_customer_attributes_silver(spark=spark, date=date_str, bronze_directory='datamart/bronze/customer_attributes/', silver_directory='datamart/silver/customer_attributes/')

# load customer financials data
print("\nLoading customer financials data...")
for date_str in dates_str_lst:
    utils.silver_processing.add_customer_financials_silver(spark=spark, date=date_str, bronze_directory='datamart/bronze/customer_financials/', silver_directory='datamart/silver/customer_financials/')

# load loan data
print("\nLoading loan data...")
for date_str in dates_str_lst:
    utils.silver_processing.add_loan_data_silver(spark=spark, date=date_str, bronze_directory='datamart/bronze/loan_data/', silver_directory='datamart/silver/loan_data/')

print("\n\n" + "-" * 50)
print("All Silver tables loaded successfully.")
print("-" * 50)

# run gold datalake backfill
print("\n\nProcessing Data to Gold tables...\n\n")

# creating gold label store
print("\nProcessing Loan Data for Gold label store...")
for date_str in dates_str_lst:
    utils.gold_processing.add_loan_data_gold_ls(spark=spark, date=date_str, silver_directory='datamart/silver/loan_data/', gold_label_store_directory='datamart/gold/label_store/')

# Feature Engineering for Feature Store
print("\nProcessing Data for Feature Store...")

for date_str in dates_str_lst:
    print(f"\nProcessing data for date: {date_str}...")

    # load clickstream data
    partition = 'silver_clickstream_data_' + date_str.replace("-", "_") + '.parquet'
    df_clickstream = utils.data_loading.load_data(spark=spark, input_directory='datamart/silver/clickstream_data/', partition=partition)
    # no further processing needed for clickstream data

    # load customer attributes data
    partition = 'silver_customer_attributes_' + date_str.replace("-", "_") + '.parquet'
    df_customer_attributes = utils.data_loading.load_data(spark=spark, input_directory='datamart/silver/customer_attributes/', partition=partition)
    # perform one-hot encoding for categorical variables
    df_customer_attributes = utils.gold_processing.one_hot_encode(df=df_customer_attributes, column="Occupation", drop_label="Other")

    # load customer financials data
    partition = 'silver_customer_financials_' + date_str.replace("-", "_") + '.parquet'
    df_customer_financials = utils.data_loading.load_data(spark=spark, input_directory='datamart/silver/customer_financials/', partition=partition)
    # perform one-hot encoding for categorical variables
    df_customer_financials = utils.gold_processing.one_hot_encode(df=df_customer_financials, column="Credit_Mix", drop_label="Unknown")
    df_customer_financials = utils.gold_processing.one_hot_encode(df=df_customer_financials, column="Payment_Behaviour", drop_label="Unknown")
    # transform Loan Type
    df_customer_financials = utils.gold_processing.encode_loan_types_with_counts(df=df_customer_financials, loan_column_name="Type_of_Loan")

    # Join all dataframes to Feature Store
    print("\nJoining current partitions to Feature Store...")

    # Step 1: Prepare Attributes Table by renaming columns
    df_attributes_renamed = df_customer_attributes \
        .withColumnRenamed("snapshot_date", "attr_effective_date") \
        .withColumnRenamed("Customer_ID", "attr_Customer_ID")

    # Step 2: Join Clickstream with Attributes (As-Of Join)
    window_attr = Window.partitionBy(F.col("cs.Customer_ID"), F.col("cs.snapshot_date")).orderBy(F.col("attr.attr_effective_date").desc())

    df_cs_attr = df_clickstream.alias("cs") \
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
    print("Schema after joining clickstream with attributes:")
    df_cs_attr.printSchema()

    # Step 3: Prepare Aliases for Financials Table
    df_financials_renamed = df_customer_financials \
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
    print(f"Total records in feature store: {df_feature_store.count()}")
    print("Schema of the final feature store:")
    df_feature_store.printSchema()

    # Ensure that gold feature store directory exists 
    os.makedirs(os.path.dirname("datamart/gold/feature_Store/"), exist_ok=True)

    # Save the cleaned data to the gold label store as parquet
    partition_name = "gold_feature_store_" + date_str.replace('-','_') + '.parquet'
    gold_feature_store_filepath = os.path.join("datamart/gold/feature_store/", partition_name)

    df_feature_store.write.mode("overwrite").parquet(gold_feature_store_filepath)
    print(f"Successfully processed and saved data to {gold_feature_store_filepath}.")


print("\n\n" + "-" * 50)
print("All Gold tables loaded successfully.")
print("-" * 50)
print("\n\n" + "-" * 50)
print("Data processing pipeline completed successfully.")
print("-" * 50)
