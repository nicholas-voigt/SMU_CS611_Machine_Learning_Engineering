import random


# -------------- Default Training Values --------------
DEFAULT_TRAINING_VALUES = {
    'start_date': '2023-01-01', # Start date for total data available for training, validation, and testing
    'end_date': '2024-06-01', # End date for total data available for training, validation, and testing
    'oot': 3,  # Number of out-of-time validation periods (each period is one month)
}


# ------------------ GBT Classifier -------------------
from pyspark.ml.classification import GBTClassifier

def get_gbt_classifier():
    """
    Returns a GBTClassifier instance and hyper parameter tuning set.
    """
    gbt_classifier = GBTClassifier(labelCol='label', featuresCol='features', seed=42)

    # Self defined hyperparameter grid for Random search
    random_grid = [
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

    return gbt_classifier, random_grid


# ------------------ Logistic Regression -------------------
from pyspark.ml.classification import LogisticRegression

def get_log_reg_classifier():
    """
    Returns a LogisticRegression instance and hyper parameter tuning set.
    """
    lr_classifier = LogisticRegression(labelCol='label', featuresCol='features', maxIter=100, elasticNetParam=0.0)

    # Randomly sample combinations
    random_grid = [
        {
            lr_classifier.regParam: random.choice([0.01, 0.02, 0.05, 0.1, 0.2]),
            lr_classifier.elasticNetParam: random.choice([0.0, 0.5, 1.0]),
            lr_classifier.maxIter: random.choice([100, 200, 500, 1000]),
            lr_classifier.threshold: random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        }
        for _ in range(100)  # x random combinations
    ]

    return lr_classifier, random_grid

