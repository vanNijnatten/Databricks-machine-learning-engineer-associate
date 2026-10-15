"""Common classification and regression metrics (exam objectives S3.11, S3.12).

Each metric is computed with scikit-learn for in-memory data and with Spark ML evaluators for
prediction DataFrames produced by a fitted Spark model.
"""

import numpy as np
from numpy.typing import ArrayLike
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.sql import DataFrame
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


def classification_metrics(y_true: ArrayLike, y_pred: ArrayLike, y_proba: ArrayLike) -> dict[str, float]:
    """Binary classification metrics; ``y_proba`` is the predicted probability of the positive class."""
    # --8<-- [start:classification_metrics]
    return {
        "accuracy": accuracy_score(y_true, y_pred),  # share of correct predictions
        "precision": precision_score(y_true, y_pred),  # of predicted positives, how many are right
        "recall": recall_score(y_true, y_pred),  # of real positives, how many we found
        "f1": f1_score(y_true, y_pred),  # harmonic mean of precision and recall
        "log_loss": log_loss(y_true, y_proba),  # penalises confident wrong probabilities
        "roc_auc": roc_auc_score(y_true, y_proba),  # ranking quality over all thresholds
        "pr_auc": average_precision_score(y_true, y_proba),  # precision-recall area, for rare positives
    }
    # --8<-- [end:classification_metrics]


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """RMSE, MAE and R-squared."""
    # --8<-- [start:regression_metrics]
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),  # same unit as the target; punishes big errors
        "mae": mean_absolute_error(y_true, y_pred),  # same unit as the target; robust to outliers
        "r2": r2_score(y_true, y_pred),  # share of variance explained; 1 is perfect, 0 = predicting the mean
    }
    # --8<-- [end:regression_metrics]


def spark_classification_metrics(predictions: DataFrame, label: str = "label") -> dict[str, float]:
    """Binary metrics from a Spark predictions DataFrame (``prediction`` and ``probability`` columns)."""
    # --8<-- [start:spark_classification]
    binary = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol="probability")
    multiclass = MulticlassClassificationEvaluator(labelCol=label, predictionCol="prediction")
    return {
        "roc_auc": binary.evaluate(predictions, {binary.metricName: "areaUnderROC"}),
        "pr_auc": binary.evaluate(predictions, {binary.metricName: "areaUnderPR"}),
        "accuracy": multiclass.evaluate(predictions, {multiclass.metricName: "accuracy"}),
        # F1 of the positive class; plain "f1" is the weighted average over both classes
        "f1": multiclass.evaluate(
            predictions,
            {multiclass.metricName: "fMeasureByLabel", multiclass.metricLabel: 1.0},
        ),
        "log_loss": multiclass.evaluate(predictions, {multiclass.metricName: "logLoss"}),
    }
    # --8<-- [end:spark_classification]


def spark_regression_metrics(predictions: DataFrame, label: str = "label") -> dict[str, float]:
    """RMSE, MAE and R-squared from a Spark predictions DataFrame."""
    # --8<-- [start:spark_regression]
    evaluator = RegressionEvaluator(labelCol=label, predictionCol="prediction")
    return {name: evaluator.evaluate(predictions, {evaluator.metricName: name}) for name in ("rmse", "mae", "r2")}
    # --8<-- [end:spark_regression]


def as_float_dict(metrics: dict[str, float]) -> dict[str, float]:
    """Convert numpy scalars to plain floats (handy before logging metrics to MLflow)."""
    return {name: float(np.asarray(value)) for name, value in metrics.items()}
