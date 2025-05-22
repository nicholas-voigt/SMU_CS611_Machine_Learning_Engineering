# This file runs the entire data preprocessing pipeline.

import os
import glob
from datetime import datetime
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.bronze_processing
import utils.silver_processing
import utils.data_loading


# Initialize SparkSession
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
print("\n\nProcessing Data to Silver tables...")

# load clickstream data
print("Loading clickstream data...")
for date_str in dates_str_lst:
    utils.silver_processing.add_clickstream_data_silver(spark=spark, date=date_str, bronze_directory='datamart/bronze/clickstream_data/', silver_directory='datamart/silver/clickstream_data/')

# load customer attributes data
print("Loading customer attributes data...")
for date_str in dates_str_lst:
    utils.silver_processing.add_customer_attributes_silver(spark=spark, date=date_str, bronze_directory='datamart/bronze/customer_attributes/', silver_directory='datamart/silver/customer_attributes/')

# load customer financials data
print("Loading customer financials data...")
for date_str in dates_str_lst:
    utils.silver_processing.add_customer_financials_silver(spark=spark, date=date_str, bronze_directory='datamart/bronze/customer_financials/', silver_directory='datamart/silver/customer_financials/')

# load loan data
print("Loading loan data...")
for date_str in dates_str_lst:
    utils.silver_processing.add_loan_data_silver(spark=spark, date=date_str, bronze_directory='datamart/bronze/loan_data/', silver_directory='datamart/silver/loan_data/')

print("\n\n" + "-" * 50)
print("All Silver tables loaded successfully.")
print("-" * 50)

# run gold datalake backfill
print("\n\nProcessing Data to Gold tables...")

# folder_path = gold_label_store_directory
# files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
# df = spark.read.option("header", "true").parquet(*files_list)
# print("row_count:",df.count())


# creating pipeline summary
print("Creating pipeline summary...")
# df = utils.data_loading.load_data(spark=spark, input_directory='datamart/silver/', partition=None)

# print("row_count:", df.count())
# df.show(5)
