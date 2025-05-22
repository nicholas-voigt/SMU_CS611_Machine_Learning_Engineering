import os
import re 

import pyspark
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col, when, lit, regexp_count
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from utils.data_loading import load_data


def add_loan_data_gold_ls(spark: SparkSession, date: str, silver_directory: str, gold_label_store_directory: str, dpd=30, mob=6):
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
    partition = 'silver_loan_data_' + date.replace("-", "_") + '.parquet'
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
    os.makedirs(os.path.dirname(gold_label_store_directory), exist_ok=True)

    # Save the cleaned data to the gold label store as parquet
    partition_name = "gold_label_store_" + date.replace('-','_') + '.parquet'
    gold_label_store_filepath = os.path.join(gold_label_store_directory, partition_name)

    df.write.mode("overwrite").parquet(gold_label_store_filepath)
    print(f"Successfully processed and saved data to {gold_label_store_filepath}.")

    return


def one_hot_encode(df: DataFrame, column: str, drop_label: str) -> DataFrame:
    '''
    Function to perform one-hot encoding on a specified column of a DataFrame.
    It creates new binary columns for each category in the input column.
    The new column names will be: original_column_name + "_" + category_value.
    The original column and the intermediate index column are kept in the DataFrame.
    Args:
        df (DataFrame): Input PySpark DataFrame.
        column (str): Column name of the string column to be one-hot encoded.
        drop_label (str): Label to be dropped from the one-hot encoding.
    Returns:
        DataFrame: DataFrame with the original column, the new index column, 
                   and the new one-hot encoded columns.
    '''

    # String Indexing to transform the string column into an index column
    indexed_col_name = column + "_index"
    indexer = StringIndexer(inputCol=column, outputCol=indexed_col_name, handleInvalid="skip")
    model = indexer.fit(df)
    df_with_index = model.transform(df)
    labels = model.labels  # Get the original string labels corresponding to the indices

    # If the drop_label is present in the labels, drop the corresponding column
    if drop_label in labels:
        labels.remove(drop_label)
    else:
        print(f"Warning: {drop_label} not found in labels. No column will be dropped.")    

    # Create a new DataFrame with the original column and the index column
    df_ohe = df_with_index
    for i, label in enumerate(labels):

        # Replace sequences of non-alphanumeric characters (excluding underscore) with a single underscore
        sanitized_label = re.sub(r'[^\w]+', '_', str(label))
        sanitized_label = sanitized_label.strip('_')
        
        # If sanitization results in an empty string, fallback
        if not sanitized_label:
            sanitized_label = f"category_{i}"

        ohe_column_name = f"{column}_{sanitized_label}"

        # Create the new binary (0 or 1) column for the current category
        df_ohe = df_ohe.withColumn(
            ohe_column_name,
            F.when(F.col(indexed_col_name) == float(i), 1).otherwise(0).cast(IntegerType())
        )
    
    return df_ohe


def encode_loan_types_with_counts(df: DataFrame, loan_column_name: str) -> DataFrame:
    """
    Encodes a column containing concatenated loan types into separate columns for each loan type,
    with each new column storing the count of that specific loan type.
    Args:
        df (DataFrame): The input PySpark DataFrame.
        loan_column_name (str): The name of the column containing the concatenated loan type strings.
    Returns:
        DataFrame: The DataFrame with new columns for each loan type count.
    """

    loan_types = [
        "Personal Loan", "Student Loan", "Payday Loan", "Auto Loan", "Home Equity Loan", 
        "Mortgage Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Not Specified"
    ]

    df_processed = df

    for loan_type in loan_types:
        # Sanitize the loan_type to make it suitable for a column name
        sanitized_loan_type = re.sub(r'[^\w]+', '_', loan_type)
        sanitized_loan_type = sanitized_loan_type.strip('_')
        if not sanitized_loan_type: # Fallback for purely special character names
            sanitized_loan_type = f"loan_type_{loan_types.index(loan_type)}"
            
        new_col_name = f"LoanType_{sanitized_loan_type}_Count"

        # Count occurrences of the loan_type string.
        df_processed = df_processed.withColumn(
            new_col_name,
            when(col(loan_column_name).isNotNull(), 
                 regexp_count(col(loan_column_name), lit(loan_type))
            ).otherwise(0).cast(IntegerType())
        )
        
    return df_processed


