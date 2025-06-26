import random

from pyspark.ml.tuning import ParamGridBuilder


# ------------------ GBT Classifier -------------------
from pyspark.ml.classification import GBTClassifier

gbt_classifier = GBTClassifier(labelCol='label', featuresCol='features', seed=42)

# Self defined hyperparameter grid for Random search
random_grid_gbt = [
    {
        gbt_classifier.maxDepth: random.choice([3, 4, 5, 6, 7, 8]),
        gbt_classifier.maxIter: random.choice([10, 20, 30, 40, 50]),
        gbt_classifier.stepSize: random.choice([0.05, 0.1, 0.2]),
        gbt_classifier.minInstancesPerNode: random.choice([1, 2, 5, 10]),
        gbt_classifier.minInfoGain: random.choice([0.0, 0.01, 0.05]),
        gbt_classifier.subsamplingRate: random.choice([0.6, 0.8, 1.0]),
        gbt_classifier.featureSubsetStrategy: random.choice(['auto', 'sqrt', 'log2', 'onethird'])
    }
    for _ in range(100)  # x random combinations
]

# Convert to ParamGridBuilder format
paramGrid = ParamGridBuilder()
for params in random_grid_gbt:
    paramGrid = paramGrid.addGrid(gbt_classifier.maxDepth, [params[gbt_classifier.maxDepth]]) \
                        .addGrid(gbt_classifier.maxIter, [params[gbt_classifier.maxIter]]) \
                        .addGrid(gbt_classifier.stepSize, [params[gbt_classifier.stepSize]]) \
                        .addGrid(gbt_classifier.minInstancesPerNode, [params[gbt_classifier.minInstancesPerNode]]) \
                        .addGrid(gbt_classifier.minInfoGain, [params[gbt_classifier.minInfoGain]]) \
                        .addGrid(gbt_classifier.subsamplingRate, [params[gbt_classifier.subsamplingRate]]) \
                        .addGrid(gbt_classifier.featureSubsetStrategy, [params[gbt_classifier.featureSubsetStrategy]])
paramGrid_GBT = paramGrid.build()

gbt = {
    'model': gbt_classifier,
    'param_grid': paramGrid_GBT
}


# ------------------ Logistic Regression -------------------
from pyspark.ml.classification import LogisticRegression

lr_classifier = LogisticRegression(labelCol='label', featuresCol='features', maxIter=100, elasticNetParam=0.0)

# Randomly sample combinations
random_grid_lr = [
    {
        lr_classifier.regParam: random.choice([0.01, 0.02, 0.05, 0.1, 0.2]),
        lr_classifier.elasticNetParam: random.choice([0.0, 0.5, 1.0]),
        lr_classifier.maxIter: random.choice([100, 200, 500, 1000]),
        lr_classifier.threshold: random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    }
    for _ in range(100)  # x random combinations
]

# Convert to ParamGridBuilder format
paramGrid = ParamGridBuilder()
for params in random_grid_lr:
    paramGrid = paramGrid.addGrid(lr_classifier.regParam, [params[lr_classifier.regParam]]) \
                        .addGrid(lr_classifier.elasticNetParam, [params[lr_classifier.elasticNetParam]]) \
                        .addGrid(lr_classifier.maxIter, [params[lr_classifier.maxIter]]) \
                        .addGrid(lr_classifier.threshold, [params[lr_classifier.threshold]])
paramGrid_LR = paramGrid.build()

logreg = {
    'model': lr_classifier,
    'param_grid': paramGrid_LR
}