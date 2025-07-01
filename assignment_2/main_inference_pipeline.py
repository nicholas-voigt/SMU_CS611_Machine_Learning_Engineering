import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from scripts.ml_processing.model_monitoring import get_population_metrics, evaluate_historic_with_labels
from configs.data import model_registry_dir, gold_data_dirs, model_prediction_dir, training_data_dir
from utils.validators import validate_date, build_partition_name, pyspark_info
from utils.data import load_data, load_metrics


# Initialize SparkSession
print("Initializing Spark session & generating snapshot dates...")
spark = SparkSession.builder \
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



#### MAIN LOOP OVER ALL DATES ####
for date_str in dates_str_lst:
    print(f"\nProcessing date: {date_str}")

    # Load the best model from the model registry
    print(f"\nLoading Champion model from model registry...")
    model_path = os.path.join(model_registry_dir, 'best_model')
    model = PipelineModel.load(model_path)

    # Load data for inference
    print(f"Loading data for inference on {date_str}...")
    partition = build_partition_name('gold', 'feature_store', date_str, 'parquet')
    df = load_data(spark, gold_data_dirs['feature_store'], partition)

    # Perform inference
    print(f"Running inference on data for {date_str}...")
    predictions = model.transform(df)

    # Log the schema of predictions
    # print(f"Predictions schema: {predictions.schema}")
    # pyspark_info(predictions)

    # Save predictions to the model prediction directory
    os.makedirs(model_prediction_dir, exist_ok=True)
    output_partition = build_partition_name('', 'model_prediction', date_str, 'parquet')
    output_path = os.path.join(model_prediction_dir, output_partition)
    print(f"Saving predictions to {output_path}...")
    predictions.write.mode('overwrite').parquet(output_path)




    ### Monitoring & Evaluation ###
    print(f"\nRunning model monitoring & evaluation for {date_str}...")

    # Load predictions from the model prediction directory
    print(f"Loading Champion Model Predictions for {date_str}...")
    partition = build_partition_name('', 'model_prediction', date_str, 'parquet')
    pred = load_data(spark, model_prediction_dir, partition)

    # Loading training data for reference
    print(f"Loading training data for reference...")
    train_data = load_data(spark, training_data_dir, 'training.parquet')

    # Load top features from the model
    metadata = load_metrics(os.path.join(model_registry_dir, 'best_model', 'best_model_metadata.json'))
    top_features = [fe_tuple[0] for fe_tuple in metadata.get('top_features', [])]
    print(f"Top features used in the model: {top_features}")

    # Calculate Population Metrics on inference date
    metrics = get_population_metrics(pred, train_data, top_features)

    # Save metrics as JSON
    log_dir = os.path.join(model_prediction_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_partition = build_partition_name('', 'prediction_log', date_str, 'json')
    print(f"Saving population metrics for {date_str}...")
    with open(file=os.path.join(log_dir, log_partition), mode="w") as f:
        json.dump(metrics, f, indent=4)

    # Evaluate historic performance with labels that became available now
    print(f"Evaluating historic performance with labels available at {date_str}...")

    label_partition = build_partition_name('gold', 'label_store', date_str, 'parquet')
    metrics, pred_date = {}, None
    if os.path.exists(os.path.join(gold_data_dirs['label_store'], label_partition)):
        print(f"Loading labels for {date_str}...")
        labels = load_data(spark, gold_data_dirs['label_store'], label_partition)

        if labels and labels.count() > 0:
            pred_date = (datetime.strptime(date_str, "%Y-%m-%d") - relativedelta(months=6)).strftime("%Y-%m-%d")
            print(f"Loading predictions for {pred_date}...")
            pred_partition = build_partition_name('', 'model_prediction', pred_date, 'parquet')
            pred = load_data(spark, model_prediction_dir, pred_partition)

            metrics = evaluate_historic_with_labels(predictions=pred, labels=labels)
        else:
            print(f"No labels found for {date_str}. Skipping evaluation.")
    else:
        print(f"No labels available for {date_str}. Skipping evaluation.")


    # add metrics to already existing logs
    if metrics and pred_date:
        partition_name = build_partition_name('', 'prediction_log', pred_date, 'json')
        pop_metrics = load_metrics(os.path.join(model_prediction_dir, 'logs', partition_name))
        with open(file=os.path.join(model_prediction_dir, 'logs', partition_name), mode="w") as f:
            json.dump({'evaluation_metrics': metrics, 'population_metrics': pop_metrics}, f, indent=4)

    print(f"Model monitoring and evaluation completed for {date_str}.\n")
    
# Stop Spark session
spark.stop()
print(f"\n\nModel inference and monitoring completed for all dates.")