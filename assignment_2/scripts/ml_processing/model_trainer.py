import os
import argparse
import tqdm

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import mlflow
import mlflow.spark

from configs.data import training_data_dir
from configs.models import gbt, logreg


if __name__ == "__main__":
    # Parse command line arguments to define model type
    parser = argparse.ArgumentParser(description='Train a machine learning model.')
    parser.add_argument('--model_type', type=str, choices=['gbt', 'logreg'], required=True, help='Type of model to train: gbt for Gradient Boosted Trees or logreg for Logistic Regression.')
    args = parser.parse_args()
    model_type = args.model_type

    # Initialize Spark session
    sparksession = SparkSession.builder \
        .appName("Model Trainer") \
        .master("local[*]") \
        .getOrCreate()
    sparksession.sparkContext.setLogLevel("ERROR")

    # Load all datasets from the training directory
    print(f"\nLoading datasets from {training_data_dir}...")
    datasets = {}
    for sets in os.listdir(training_data_dir):
        if sets.endswith('.parquet'):
            dataset_name = sets.split('.')[0]  # Remove the .parquet extension
            datasets[dataset_name] = sparksession.read.parquet(os.path.join(training_data_dir, sets))
        else:
            print(f"Skipping unsupported file format: {sets}")

    for name, df in datasets.items():
        print(f"Dataset: {name} with {df.count()} rows.")
    
    # Prepare train, validation and test datasets
    train_df = datasets['training']
    validation_df = datasets['validation']
    test_df = datasets['test']

    # Assemble features into a single vector column
    assembler = VectorAssembler(
        inputCols=[col for col in train_df.columns if col not in ['Customer_ID', 'label', 'snapshot_date']], 
        outputCol='features'
    )
    
    # Load Model Configuration
    model, param_grid = (gbt['model'], gbt['param_grid']) if model_type == 'gbt' else (logreg['model'], logreg['param_grid'])

    # Prepare pipeline
    pipeline = Pipeline(stages=[assembler, model])
    evaluator = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC')

    # Fit and tune the model
    print("\nStarting model training and hyperparameter tuning...")
    results = []
    for params in tqdm.tqdm(param_grid, desc="Training models", unit="model"):
        # Set parameters for this run
        for param, value in params.items():
            model._set(**{param.name: value})

        # Fit pipeline on train_df
        model = pipeline.fit(train_df)
        
        # Evaluate on train and validation data
        train_predictions = model.transform(train_df)
        train_auc = evaluator.evaluate(train_predictions)
        val_predictions = model.transform(validation_df)
        val_auc = evaluator.evaluate(val_predictions)
        
        # Store results
        results.append({
            'params': {param.name: value for param, value in params.items()},
            'metrics': {'train_auc': train_auc, 'val_auc': val_auc},
            'model': model
        })

    # Find the best model by validation AUC & evaluating on test and OOT datasets
    print("\nExtracting the best model based on validation AUC & evaluating on test & oot data...")
    best_result = max(results, key=lambda x: x['metrics']['val_auc'])
    best_model = best_result['model']

    test_predictions = best_model.transform(test_df)
    best_result['metrics']['test_auc'] = evaluator.evaluate(test_predictions)

    for i in range(1, len(datasets) - 2):
        oot_df = datasets[f'oot_{i}']
        oot_predictions = best_model.transform(oot_df)
        best_result['metrics'][f'oot_{i}_auc'] = evaluator.evaluate(oot_predictions)

    # Extract top 10 features based on feature importance
    if model_type == 'gbt':
        feature_importances = best_model.stages[-1].featureImportances.toArray()
    else:
        feature_importances = best_model.stages[-1].coefficients.toArray()
    top_features = sorted(zip(assembler.getInputCols(), feature_importances), key=lambda x: x[1], reverse=True)[:10]

    # Log results, features and model to mlflow
    print("\nTraining completed. Best model:")
    for set, metrics in best_result['metrics'].items():
        print(f"{set}: {metrics}")
    print("Top 10 features:")
    for feature, importance in top_features:
        print(f"{feature}: {importance}")

    with mlflow.start_run():
        # Log parameters and metrics
        mlflow.log_params(best_result['params'])
        for metric, score in best_result['metrics'].items():
            mlflow.log_metric(metric, score)
        
        # Log the model
        mlflow.spark.log_model(
            best_model, "model", 
            registered_model_name=f"{model_type.upper()}_Classifier_Model"
        )
    
    # Stop Spark session
    sparksession.stop()
    print("\n\nModel training and logging completed successfully.\n\n")