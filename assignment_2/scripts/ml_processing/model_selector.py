import os
import json

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

from configs.data import model_registry_dir


def load_metrics(metrics_path):
    with open(metrics_path, "r") as f:
        return json.load(f)


def select_best_model(models_dir, criterion="val_auc"):
    best_score = float('-inf')
    best_model = None
    best_info = None

    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name, "model")
        metrics_path = os.path.join(models_dir, model_name, "metrics.json")
        if not os.path.exists(model_path) or not os.path.exists(metrics_path):
            continue

        # Load model and metrics
        model = PipelineModel.load(model_path)
        metrics = load_metrics(metrics_path)

        # Use the specified criterion (e.g., val_auc, test_auc, etc.)
        score = metrics.get(criterion)
        if score is not None and score > best_score:
            best_score = score
            best_model = model
            best_info = {
                "model_name": model_name,
                "metrics": metrics
            }

    if best_model and best_info:
        print(f"Best model: {best_info['model_name']}")
        print("Metrics:", best_info["metrics"])
        return best_model, best_info
    else:
        print("No valid models found.")
        return None, None


if __name__ == "__main__":
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Training Data Preparation") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Load all models from the model registry directory
    print(f"\nLoading models from {model_registry_dir}...")
    if not os.path.exists(model_registry_dir):
        raise FileNotFoundError(f"Model registry directory {model_registry_dir} does not exist.")
    
    # Iterate through all model directories and select the best model based on validation AUC
    best_model, best_info = select_best_model(model_registry_dir)

    # Save the best model if found in best registry
    if best_model and best_info:
        best_model.save(os.path.join(model_registry_dir, "best_model"))
        with open(os.path.join(model_registry_dir, "best_model", "metrics.json"), "w") as f:
            json.dump(best_info["metrics"], f, indent=4)
    
        print(f"Best model saved to {os.path.join(model_registry_dir, 'best_model')}")
    else:
        print("No best model found to save.")
    
    # Stop the Spark session
    spark.stop()
    print("Spark session stopped.")