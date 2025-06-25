import os
import random
import tqdm

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import mlflow
import mlflow.spark

from configs.data import training_data_dir


if __name__ == "__main__":
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Training Data Preparation") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Load all datasets from the training directory
    print(f"\nLoading datasets from {training_data_dir}...")
    datasets = {}
    for sets in os.listdir(training_data_dir):
        if sets.endswith('.parquet'):
            dataset_name = sets.split('.')[0]  # Remove the .parquet extension
            datasets[dataset_name] = spark.read.parquet(os.path.join(training_data_dir, sets))
        else:
            print(f"Skipping unsupported file format: {sets}")

    for name, df in datasets.items():
        print(f"Dataset: {name} with {df.count()} rows.")
    
    # Prepare train, validation and test datasets for GBT Classifier
    train_df = datasets['training']
    validation_df = datasets['validation']
    test_df = datasets['test']

    # Assemble features into a single vector column
    assembler = VectorAssembler(
        inputCols=[col for col in train_df.columns if col not in ['Customer_ID', 'label', 'snapshot_date']], 
        outputCol='features'
    )

    # Define the GBT Classifier    
    gbt = GBTClassifier(labelCol='label', featuresCol='features', seed=42)

    # Define possible values
    maxDepths = [3, 4, 5, 6, 7, 8]
    maxIters = [10, 20, 30, 40, 50]
    stepSizes = [0.05, 0.1, 0.2]
    minInstancesPerNode = [1, 2, 5, 10]
    minInfoGain = [0.0, 0.01, 0.05]
    subsamplingRates = [0.6, 0.8, 1.0]
    featureSubsetStrategies = ['auto', 'sqrt', 'log2', 'onethird']

    # Randomly sample combinations
    random_grid = [
        {
            gbt.maxDepth: random.choice(maxDepths),
            gbt.maxIter: random.choice(maxIters),
            gbt.stepSize: random.choice(stepSizes),
            gbt.minInstancesPerNode: random.choice(minInstancesPerNode),
            gbt.minInfoGain: random.choice(minInfoGain),
            gbt.subsamplingRate: random.choice(subsamplingRates),
            gbt.featureSubsetStrategy: random.choice(featureSubsetStrategies)
        }
        for _ in range(100)  # x random combinations
    ]

    # Convert to ParamGridBuilder format
    paramGrid = ParamGridBuilder()
    for params in random_grid:
        paramGrid = paramGrid.addGrid(gbt.maxDepth, [params[gbt.maxDepth]]) \
                            .addGrid(gbt.maxIter, [params[gbt.maxIter]]) \
                            .addGrid(gbt.stepSize, [params[gbt.stepSize]]) \
                            .addGrid(gbt.minInstancesPerNode, [params[gbt.minInstancesPerNode]]) \
                            .addGrid(gbt.minInfoGain, [params[gbt.minInfoGain]]) \
                            .addGrid(gbt.subsamplingRate, [params[gbt.subsamplingRate]]) \
                            .addGrid(gbt.featureSubsetStrategy, [params[gbt.featureSubsetStrategy]])
    paramGrid = paramGrid.build()


    # Prepare pipeline
    pipeline = Pipeline(stages=[assembler, gbt])
    evaluator = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC')

    # Fit and tune the model
    print("\nStarting model training and hyperparameter tuning...")
    results = []
    for params in tqdm.tqdm(random_grid, desc="Training models", unit="model"):
        # Set parameters for this run
        for param, value in params.items():
            gbt._set(**{param.name: value})
        
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
    
    # Saving the best model using MLflow
    print("\nSaving model and logging results in MLflow...")
    mlflow.set_experiment("GBT_Classifier_Training")
    with mlflow.start_run():
        # Log parameters and metrics
        mlflow.log_params(best_result['params'])
        for metric, score in best_result['metrics'].items():
            mlflow.log_metric(metric, score)
        # Log the model
        mlflow.spark.log_model(
            best_model, "model", 
            registered_model_name="GBT_Classifier_Model"
            )

    print("\nTraining completed. Best model parameters and metrics:")
    print(best_result['params'])
    print(best_result['metrics'])