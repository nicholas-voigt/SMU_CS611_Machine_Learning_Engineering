# The following functions are used to process the data in the silver layer. They are called in the main.py file.

import os
from datetime import datetime
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, StringType, DateType, DecimalType, FloatType, BooleanType
from assignment_2.scripts.data_processing.data_loading import load_data


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


def add_customer_attributes_silver(spark: SparkSession, date: str, bronze_directory: str, silver_directory: str):
    '''
    Function to process customer attributes data from bronze table to silver table. Enforces schema and data quality checks.
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
    partition = 'bronze_customer_attributes_' + date.replace("-", "_") + '.csv'
    df = load_data(spark, bronze_directory, partition)
    bronze_count = df.count()
    if df is None or bronze_count == 0:
        raise ValueError(f"No data found in bronze directory for partition {partition}.")
    
    # Data quality checks

    # drop columns name and ssn
    df = df.drop("Name", "SSN")

    # enforce schema
    column_type_map = {
        "Age": IntegerType(),
        "Occupation": StringType(),
        "Customer_ID": StringType(),
        "snapshot_date": DateType()
        }
    for column, dtype in column_type_map.items():
        df = df.withColumn(column, col(column).cast(dtype))

    # Handle corrupted values (1 of 15 categories for occupation, sanity check for age, remove if customer_id or snapshot_date are corrupted)
    df = df.filter(col("Customer_ID").isNotNull() | col("snapshot_date").isNotNull())
    df = df.withColumn("Age", F.when((col("Age") >= 0) & (col("Age") <= 100), col("Age")).otherwise(0))
    age_null_count = df.filter(col("Age") == 0).count()
    df = df.withColumn("Occupation", 
                       F.when(col("Occupation").isin([
                           "Scientist", 
                           "Media_Manager", 
                           "Musician", 
                           "Lawyer", 
                           "Teacher", 
                           "Developer", 
                           "Writer", 
                           "Architect", 
                           "Mechanic", 
                           "Entrepreneur", 
                           "Journalist", 
                           "Doctor", 
                           "Engineer", 
                           "Accountant", 
                           "Manager", 
                           "Other"
                           ]), 
                        col("Occupation")).otherwise("Other"))
    occupation_null_count = df.filter(col("Occupation") == "Other").count()
    # Remove duplicates
    df = df.dropDuplicates(["Customer_ID", "snapshot_date"])
    
    # check for row count after cleaning
    row_difference = df.count() - bronze_count
    if any([row_difference, age_null_count, occupation_null_count]) > 0:
        print(f"Warning: Cleaning resulted in {row_difference} rows removed, {age_null_count} age nulls, and {occupation_null_count} occupation nulls for partition {partition}.")

    # Ensure that silver directory exists 
    os.makedirs(os.path.dirname(silver_directory), exist_ok=True)

    # Save the cleaned data to the silver directory as parquet
    silver_partition_name = 'silver_customer_attributes_' + date.replace("-", "_") + '.parquet'
    silver_filepath = os.path.join(silver_directory, silver_partition_name)

    df.write.mode("overwrite").parquet(silver_filepath)
    print(f"Successfully processed and saved data to {silver_filepath}.")

    return


def add_customer_financials_silver(spark: SparkSession, date: str, bronze_directory: str, silver_directory: str):
    '''
    Function to process customer financials data from bronze table to silver table. Enforces schema and data quality checks.
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
    partition = 'bronze_customer_financials_' + date.replace("-", "_") + '.csv'
    df = load_data(spark, bronze_directory, partition)
    bronze_count = df.count()
    if df is None or bronze_count == 0:
        raise ValueError(f"No data found in bronze directory for partition {partition}.")
    
    # Data quality checks

    # enforce schema
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": DecimalType(18, 2),
        "Monthly_Inhand_Salary": DecimalType(18, 2),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": DecimalType(18, 2),
        "Credit_Utilization_Ratio": DecimalType(4, 2),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": BooleanType(),
        "Total_EMI_per_month": DecimalType(18, 2),
        "Amount_invested_monthly": DecimalType(18, 2),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": DecimalType(18, 2),
        "snapshot_date": DateType()
        }
    
    for column, dtype in column_type_map.items():
        df = df.withColumn(column, col(column).cast(dtype))

    # Handle corrupted values, transform values
    ## identifiers have to be present
    df = df.filter(col("Customer_ID").isNotNull() | col("snapshot_date").isNotNull())

    ## enforce false for boolean values if not present
    df = df.withColumn("Payment_of_Min_Amount", F.when(col("Payment_of_Min_Amount").isNotNull(), col("Payment_of_Min_Amount")).otherwise(False))

    ## numerical values checking (> 0 for all, < 100 for interest rate and credit utilization ratio)
    df = df.withColumn("Annual_Income", F.when(col("Annual_Income") > 0, col("Annual_Income")).otherwise(0))
    df = df.withColumn("Monthly_Inhand_Salary", F.when(col("Monthly_Inhand_Salary") > 0, col("Monthly_Inhand_Salary")).otherwise(0))
    df = df.withColumn("Num_Bank_Accounts", F.when(col("Num_Bank_Accounts") > 0, col("Num_Bank_Accounts")).otherwise(0))
    df = df.withColumn("Num_Credit_Card", F.when(col("Num_Credit_Card") > 0, col("Num_Credit_Card")).otherwise(0))
    df = df.withColumn("Interest_Rate", F.when((col("Interest_Rate") > 0) & (col("Interest_Rate") < 100), col("Interest_Rate")).otherwise(0))
    df = df.withColumn("Num_of_Loan", F.when(col("Num_of_Loan") > 0, col("Num_of_Loan")).otherwise(0))
    df = df.withColumn("Delay_from_due_date", F.when(col("Delay_from_due_date") > 0, col("Delay_from_due_date")).otherwise(0))
    df = df.withColumn("Num_of_Delayed_Payment", F.when(col("Num_of_Delayed_Payment") > 0, col("Num_of_Delayed_Payment")).otherwise(0))
    df = df.withColumn("Changed_Credit_Limit", F.when(col("Changed_Credit_Limit") > 0, col("Changed_Credit_Limit")).otherwise(0))
    df = df.withColumn("Num_Credit_Inquiries", F.when(col("Num_Credit_Inquiries") > 0, col("Num_Credit_Inquiries")).otherwise(0))
    df = df.withColumn("Outstanding_Debt", F.when(col("Outstanding_Debt") > 0, col("Outstanding_Debt")).otherwise(0))
    df = df.withColumn("Credit_Utilization_Ratio", F.when((col("Credit_Utilization_Ratio") > 0) & (col("Credit_Utilization_Ratio") < 100), col("Credit_Utilization_Ratio")).otherwise(0))
    df = df.withColumn("Total_EMI_per_month", F.when(col("Total_EMI_per_month") > 0, col("Total_EMI_per_month")).otherwise(0))
    df = df.withColumn("Amount_invested_monthly", F.when(col("Amount_invested_monthly") > 0, col("Amount_invested_monthly")).otherwise(0))
    df = df.withColumn("Monthly_Balance", F.when(col("Monthly_Balance") > 0, col("Monthly_Balance")).otherwise(0))
    
    ## categorical values checking
    df = df.withColumn("Credit_Mix", 
                       F.when(col("Credit_Mix").isin([
                           "Good", 
                           "Bad", 
                           "Standard", 
                           "Unknown"
                           ]), 
                        col("Credit_Mix")).otherwise("Unknown"))
    df = df.withColumn("Payment_Behaviour",
                       F.when(col("Payment_Behaviour").isin([
                           "Low_spent_Small_value_payments", 
                           "Low_spent_Medium_value_payments", 
                           "Low_spent_Large_value_payments", 
                           "High_spent_Small_value_payments", 
                           "High_spent_Medium_value_payments", 
                           "High_spent_Large_value_payments"
                           ]), 
                        col("Payment_Behaviour")).otherwise("Unknown"))
    
    ## Transform credit history age to no of months 
    years_col = F.coalesce(
        F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Year(s)?", 1).cast(IntegerType()), 
        F.lit(0)
    )
    months_col = F.coalesce(
        F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Month(s)?", 1).cast(IntegerType()), 
        F.lit(0)
    )
    df = df.withColumn("Credit_History_Age", ((years_col * 12) + months_col).cast(IntegerType()))

    # Remove duplicates
    df = df.dropDuplicates(["Customer_ID", "snapshot_date"])
    
    # check for row count after cleaning
    if bronze_count - df.count() != 0:
        print(f"Warning: Cleaning resulted in {bronze_count - df.count()} rows removed for partition {partition}.")

    # Ensure that silver directory exists 
    os.makedirs(os.path.dirname(silver_directory), exist_ok=True)

    # Save the cleaned data to the silver directory as parquet
    silver_partition_name = 'silver_customer_financials_' + date.replace("-", "_") + '.parquet'
    silver_filepath = os.path.join(silver_directory, silver_partition_name)

    df.write.mode("overwrite").parquet(silver_filepath)
    print(f"Successfully processed and saved data to {silver_filepath}.")

    return


def add_loan_data_silver(spark: SparkSession, date: str, bronze_directory: str, silver_directory: str):
    '''
    Function to process loan data from bronze table to silver table. Enforces schema and data quality checks.
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
    partition = 'bronze_loan_data_' + date.replace("-", "_") + '.csv'
    df = load_data(spark, bronze_directory, partition)
    bronze_count = df.count()
    if df is None or bronze_count == 0:
        raise ValueError(f"No data found in bronze directory for partition {partition}.")
    
    # Data quality checks

    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # check for row count after cleaning
    if bronze_count - df.count() != 0:
        print(f"Warning: Cleaning resulted in {bronze_count - df.count()} rows removed for partition {partition}.")

    # Ensure that silver directory exists 
    os.makedirs(os.path.dirname(silver_directory), exist_ok=True)

    # Save the cleaned data to the silver directory as parquet
    silver_partition_name = 'silver_loan_data_' + date.replace("-", "_") + '.parquet'
    silver_filepath = os.path.join(silver_directory, silver_partition_name)

    df.write.mode("overwrite").parquet(silver_filepath)
    print(f"Successfully processed and saved data to {silver_filepath}.")
    
    return