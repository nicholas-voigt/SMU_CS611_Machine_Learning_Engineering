import os
import argparse

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

from utils.validators import validate_date, build_partition_name
from utils.data import load_data
from configs.data import model_registry_dir, gold_data_dirs, model_prediction_dir

if __name__ == "__main__":
    # Parse command line arguments to define model type
    parser = argparse.ArgumentParser(description='Train a machine learning model.')
    parser.add_argument('--date', type=str, required=True, help='Date for which to run the model inference (YYYY-MM-DD).')
    args = parser.parse_args()
    inference_date = str(validate_date(args.date))

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Model Trainer") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Load the best model from the model registry
    print(f"\nLoading Champion model from model registry...")
    model_path = os.path.join(model_registry_dir, 'best_model')
    model = PipelineModel.load(model_path)

    # Load data for inference
    print(f"Loading data for inference on {inference_date}...")
    partition = build_partition_name('gold', 'feature_store', inference_date, 'parquet')
    df = load_data(spark, gold_data_dirs['feature_store'], partition)

    # Perform inference
    print(f"Running inference on data for {inference_date}...")
    predictions = model.transform(df)

    # Save predictions to the model prediction directory
    os.makedirs(model_prediction_dir, exist_ok=True)
    output_partition = build_partition_name('', 'model_prediction', inference_date, 'parquet')
    output_path = os.path.join(model_prediction_dir, output_partition)
    print(f"Saving predictions to {output_path}...")
    predictions.write.mode('overwrite').parquet(output_path)

    # Stop Spark session
    spark.stop()
    print(f"\n\nModel inference completed for {inference_date}. Predictions saved to {output_path}.\n\n")