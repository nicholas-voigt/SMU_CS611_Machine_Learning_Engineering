import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import DataType, StringType, IntegerType, FloatType, DateType, DecimalType, BooleanType
from pyspark.sql.functions import regexp_extract, col

from .validators import validate_file_path


def load_data(spark: SparkSession, input_directory: str, partition: str | None) -> DataFrame:
    """
    Load data from a CSV file into a Spark DataFrame.
    Args:
        spark (SparkSession): The Spark session object.
        input_directory (str): The directory to the input files. Files should be in CSV or Parquet format.
        partition (str): A specific partition to load. If not provided, all data in the given directory will be loaded.
    Returns:
        DataFrame: A Spark DataFrame containing the loaded data.
    """
    # Check if input directory exists
    if not os.path.exists(input_directory):
        raise FileNotFoundError(f"Input directory {input_directory} does not exist.")
    # If partition is provided, check if file exists and load it
    if partition:
        file_path = validate_file_path(os.path.join(input_directory, partition))
        # Load CSV or parquet file based on the file extension
        if file_path.endswith('.parquet'):
            df = spark.read.parquet(file_path)
        elif file_path.endswith('.csv'):
            df = spark.read.csv(file_path, header=True, inferSchema=True)
        else:
            raise ValueError(f"Unsupported file format for {file_path}. Only CSV and Parquet are supported.")
        print(f"Loaded data from {file_path}. Row count: {df.count()}")
        return df
    
    # If no partition is provided, load all data in the directory
    else:
        # check which file types are in the directory
        file_types = set()
        for file in os.listdir(input_directory):
            if file.endswith('.csv'):
                file_types.add('csv')
            elif file.endswith('.parquet'):
                file_types.add('parquet')
            else:
                raise ValueError(f"Unsupported file format in {input_directory}. Only CSV and Parquet are supported.")
        # If both file types are present, raise an error
        if len(file_types) > 1:
            raise ValueError(f"Multiple file types found in {input_directory}. Please specify a partition.")
        
        # Load files based on the file type
        if 'parquet' in file_types:
            all_files_path = os.path.join(input_directory, "*.parquet")
            print(f"Loading parquet files with wildcard: {all_files_path}")
            df = spark.read.parquet(all_files_path)
        else:
            all_files_path = os.path.join(input_directory, "*.csv")
            print(f"Loading csv files with wildcard: {all_files_path}")
            df = spark.read.csv(all_files_path, header=True, inferSchema=True)
        
        print(f"Loaded data from all files from {input_directory}. Row count: {df.count()}")
        return df


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

