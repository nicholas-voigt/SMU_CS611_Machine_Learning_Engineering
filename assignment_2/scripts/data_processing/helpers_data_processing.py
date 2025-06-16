import os
import re
from datetime import datetime

from pyspark.sql import DataFrame
from pyspark.sql.types import DataType, StringType, IntegerType, FloatType, DateType, DecimalType, BooleanType
from pyspark.sql.functions import regexp_extract, col, count


def validate_date(date_str):
    """
    Validate the date string against the YYYY-MM-DD format.
    Args:
        date_str (str): The date string to validate.
    Returns:
        datetime: The validated date as a datetime object.
    """
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        raise ValueError(f"Invalid date format: {date_str}. Expected format is YYYY-MM-DD.")
    
    return datetime.strptime(date_str, "%Y-%m-%d")


def validate_file_path(file_path):
    """
    Validate if the file path exists.
    Args:
        file_path (str): The file path to validate.
    Returns:
        str: The validated file path.
    """
    if not re.match(r'^[\w\-. /]+$', file_path) or not os.path.exists(file_path):
        raise ValueError(f"Invalid file path: {file_path}")
    
    return file_path


def build_partition_name(level: str, table_name: str, date: str, type: str) -> str:
    """
    Builds a partition name for saving data based on the table name and date.
    Args:
        table_name (str): The name of the table.
        date (str): The date in YYYY-MM-DD format.
    Returns:
        str: The partition name in the format 'bronze_<table_name>_<date>.csv'.
    """
    if not re.match(r'^[a-zA-Z0-9_]+$', table_name):
        raise ValueError(f"Invalid table name: {table_name}. Only alphanumeric characters and underscores are allowed.")

    return f"{level}_{table_name}_{date.replace('-', '_')}.{type}"


def transform_data_in_column(df: DataFrame, column_name: str, dtype: DataType):
    """
    Transforms data from a specified column in a DataFrame.
    Args:
        df (DataFrame): The DataFrame to transform data in.
        column_name (str): The name of the column to transform the data in.
        dtype (DataType): The target data type to transform the column to.
    Returns:
        df (DataFrame): The DataFrame with the transformed column.
    """

    # Regex pattern depending on the data type
    if isinstance(dtype, StringType):
        pattern = r'^[a-zA-Z0-9_]+$'  # Alphanumeric and underscore
    elif isinstance(dtype, IntegerType):
        pattern = r'^\d+$'  # Digits only
    elif isinstance(dtype, FloatType) or isinstance(dtype, DecimalType):
        pattern = r'^\d+(\.\d+)?$'  # Digits with optional decimal point
    elif isinstance(dtype, BooleanType):
        pattern = r'^(true|false|1|0)$'
    elif isinstance(dtype, DateType):
        pattern = r'^\d{4}-\d{2}-\d{2}$'  # YYYY-MM-DD format
    else:
        raise ValueError(f"Unsupported data type: {dtype}")
    
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Extract the matching part of the string using regexp_extract
    df = df.withColumn(
        column_name,
        regexp_extract(col(column_name).cast(StringType()), pattern, 0)
    )

    # Cast the column to the specified data type & return the DataFrame
    return df.withColumn(column_name, df[column_name].try_cast(dtype))


def pyspark_df_info(df: DataFrame):
    """
    Prints information about a PySpark DataFrame, similar to pandas.DataFrame.info().
    Includes total number of rows, and for each column: its name, 
    the number of non-null entries, and its data type.

    Args:
        df: The PySpark DataFrame to get information about.
    """

    if not isinstance(df, DataFrame):
        print("Error: Input is not a PySpark DataFrame.")
        return

    total_rows = df.count()
    print(f"\nTotal entries: {total_rows}")
    
    num_columns = len(df.columns)
    print(f"Data columns (total {num_columns} columns):")
    
    header = f"{'#':<3} {'Column':<25} {'Non-Null Count':<18} {'Dtype':<15}"
    print(header)
    print("--- " + "-"*25 + " " + "-"*18 + " " + "-"*15)

    # get non-null counts for all columns
    agg_exprs = [count(c).alias(c) for c in df.columns]
    non_null_counts_row = df.agg(*agg_exprs).collect()[0]

    for i, (col_name, col_type) in enumerate(df.dtypes):
        non_null_count = non_null_counts_row[col_name]
        print(f"{i:<3} {col_name:<25} {non_null_count:<18} {col_type:<15}")
    print("\n")