import os
import argparse
import numpy as np
import json
from scipy.stats import ks_2samp
from dateutil.relativedelta import relativedelta
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import DoubleType

from utils.validators import validate_date, build_partition_name
from utils.data import load_data, load_metrics
from configs.data import model_prediction_dir, training_data_dir, model_registry_dir, gold_data_dirs


def calculate_psi(expected, actual, buckets=10):
    """Calculate PSI for a single feature."""
    expected_percents, _ = np.histogram(expected, bins=buckets)
    actual_percents, _ = np.histogram(actual, bins=buckets)
    expected_percents = expected_percents / len(expected)
    actual_percents = actual_percents / len(actual)
    psi = np.sum((expected_percents - actual_percents) * np.log((expected_percents + 1e-6) / (actual_percents + 1e-6)))
    return float(psi)


def calculate_ks(expected, actual):
    """Calculate KS statistic for a single feature."""
    ks_result = ks_2samp(expected, actual)
    return float(ks_result.statistic) # type: ignore


def evaluate_historic_with_labels(predictions: DataFrame, labels: DataFrame):
    """
    Load labels which are recorded on date and evaluate model performance
    of respective predictions 6 months ago.
    Args:
        predictions (DataFrame): DataFrame containing model predictions.
        labels (DataFrame): DataFrame containing true labels.
    Returns:
        dict: Dictionary containing evaluation metrics.
    """

    print(f"Joining predictions with labels & evaluating performance...")
    df = predictions.join(labels, on='customer_id', how='inner')

    evaluator = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC')
    auc = evaluator.evaluate(df)
    print(f"AUC: {auc}")
    return {"auc": auc}


def get_population_metrics(predictions: DataFrame, reference_data: DataFrame, feature_list=None):
    """    
    Load predictions and calculate population metrics like PSI and KS for each feature.
    Args:
        predictions (DataFrame): DataFrame containing model predictions.
        reference_data (DataFrame): DataFrame containing reference data for comparison.
    Returns:
        dict: Dictionary containing metrics for each feature.
    """
    # Analyze population drift
    print("Analyzing population drift...")
    if feature_list is None:
        feature_list = reference_data.columns
        feature_list = [f for f in feature_list if f not in ['Customer_ID', 'snapshot_date']]

    metric_results = {
        'feature': [],
        'psi': [],
        'ks': [],
        'mean_train': [],
        'mean_curr': [],
        'std_train': [],
        'std_curr': [],
        'label_ratio': 0.0
    }

    for feature in feature_list:
        metric_results['feature'].append(feature)
        # Collect feature values as numpy arrays from Spark DataFrames
        ref_values = reference_data.select(reference_data[feature].cast(DoubleType())).rdd.flatMap(lambda x: x).collect()
        pred_values = predictions.select(predictions[feature].cast(DoubleType())).rdd.flatMap(lambda x: x).collect()

        psi = calculate_psi(np.array(ref_values), np.array(pred_values))
        ks = calculate_ks(np.array(ref_values), np.array(pred_values))
        mean_train, mean_curr = float(np.mean(ref_values)), float(np.mean(pred_values))
        std_train, std_curr = float(np.std(ref_values)), float(np.std(pred_values))

        metric_results['psi'].append(psi)
        metric_results['ks'].append(ks)
        metric_results['mean_train'].append(mean_train)
        metric_results['mean_curr'].append(mean_curr)
        metric_results['std_train'].append(std_train)
        metric_results['std_curr'].append(std_curr)

    metric_results['label_ratio'] = (
        predictions.filter(predictions['prediction'] == 1).count() / predictions.count()
        if predictions.count() > 0 else 0.0
    )
    print("Population metrics calculated.")
    return metric_results


if __name__ == "__main__":
    # Parse command line arguments to define model type
    parser = argparse.ArgumentParser(description='Train a machine learning model.')
    parser.add_argument('--date', type=str, required=True, help='Date for which to run the model inference (YYYY-MM-DD).')
    args = parser.parse_args()
    inference_date_str = str(validate_date(args.date))

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Model Trainer") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Load predictions from the model prediction directory
    print(f"Loading Champion Model Predictions for {inference_date_str}...")
    partition = build_partition_name('', 'model_prediction', inference_date_str, 'parquet')
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
    log_partition = build_partition_name('', 'prediction_log', inference_date_str, 'json')
    print(f"Saving population metrics for {inference_date_str}...")
    with open(file=os.path.join(log_dir, log_partition), mode="w") as f:
        json.dump(metrics, f, indent=4)

    # Evaluate historic performance with labels that became available now
    print(f"Evaluating historic performance with labels available at {inference_date_str}...")

    label_partition = build_partition_name('gold', 'label_store', inference_date_str, 'parquet')
    metrics, pred_date = {}, None
    if os.path.exists(os.path.join(gold_data_dirs['label_store'], label_partition)):
        print(f"Loading labels for {inference_date_str}...")
        labels = load_data(spark, gold_data_dirs['label_store'], label_partition)

        if labels and labels.count() > 0:
            pred_date = (datetime.strptime(inference_date_str, "%Y-%m-%d") - relativedelta(months=6)).strftime("%Y-%m-%d")
            print(f"Loading predictions for {pred_date}...")
            pred_partition = build_partition_name('', 'model_prediction', pred_date, 'parquet')
            pred = load_data(spark, model_prediction_dir, pred_partition)

            metrics = evaluate_historic_with_labels(predictions=pred, labels=labels)
        else:
            print(f"No labels found for {inference_date_str}. Skipping evaluation.")
    else:
        print(f"No labels available for {inference_date_str}. Skipping evaluation.")


    # add metrics to already existing logs
    if metrics and pred_date:
        partition_name = build_partition_name('', 'prediction_log', pred_date, 'json')
        pop_metrics = load_metrics(os.path.join(model_prediction_dir, 'logs', partition_name))
        with open(file=os.path.join(model_prediction_dir, 'logs', partition_name), mode="w") as f:
            json.dump({'evaluation_metrics': metrics, 'population_metrics': pop_metrics}, f, indent=4)

    # Stop the Spark session
    spark.stop()
    print(f"\n\nModel monitoring and evaluation completed for {inference_date_str}.\n\n")