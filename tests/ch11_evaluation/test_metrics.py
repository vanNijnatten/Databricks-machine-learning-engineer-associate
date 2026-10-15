"""Classification and regression metrics in scikit-learn and Spark ML (S3.11-S3.13)."""

import numpy as np
import pandas as pd
import pytest
from pyspark.ml.functions import array_to_vector
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.metrics import (
    f1_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

from db_ml_associate.ch11_evaluation.metric_choice import Scenario, choose_metric
from db_ml_associate.ch11_evaluation.metrics import (
    classification_metrics,
    regression_metrics,
    spark_classification_metrics,
    spark_regression_metrics,
)

RNG = np.random.default_rng(0)
Y_TRUE = RNG.integers(0, 2, size=400)
Y_PROBA = np.clip(Y_TRUE * 0.35 + RNG.random(400) * 0.65, 0.01, 0.99)
Y_PRED = (Y_PROBA >= 0.5).astype(int)
R_TRUE = RNG.normal(100, 20, size=300)
R_PRED = R_TRUE + RNG.normal(0, 8, size=300)


def test_classification_metrics_match_sklearn() -> None:
    metrics = classification_metrics(Y_TRUE, Y_PRED, Y_PROBA)
    assert metrics["f1"] == pytest.approx(f1_score(Y_TRUE, Y_PRED))
    assert metrics["log_loss"] == pytest.approx(log_loss(Y_TRUE, Y_PROBA))
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(Y_TRUE, Y_PROBA))
    assert set(metrics) >= {"accuracy", "precision", "recall", "pr_auc"}


def test_regression_metrics_match_definitions() -> None:
    metrics = regression_metrics(R_TRUE, R_PRED)
    assert metrics["rmse"] == pytest.approx(np.sqrt(np.mean((R_TRUE - R_PRED) ** 2)))
    assert metrics["mae"] == pytest.approx(mean_absolute_error(R_TRUE, R_PRED))
    assert metrics["r2"] == pytest.approx(r2_score(R_TRUE, R_PRED))
    assert metrics["rmse"] >= metrics["mae"]


@pytest.mark.spark
def test_spark_classification_evaluators_agree_with_sklearn(
    spark: SparkSession,
) -> None:
    frame = pd.DataFrame(
        {
            "label": Y_TRUE.astype(float),
            "prediction": Y_PRED.astype(float),
            "p1": Y_PROBA,
        }
    )
    predictions = spark.createDataFrame(frame).withColumn(
        "probability", array_to_vector(F.array(1 - F.col("p1"), F.col("p1")))
    )  # the vector column a fitted Spark classifier produces
    metrics = spark_classification_metrics(predictions, label="label")
    expected = classification_metrics(Y_TRUE, Y_PRED, Y_PROBA)
    assert metrics["roc_auc"] == pytest.approx(expected["roc_auc"], abs=1e-3)
    assert metrics["f1"] == pytest.approx(expected["f1"])
    assert metrics["log_loss"] == pytest.approx(expected["log_loss"], rel=1e-6)


@pytest.mark.spark
def test_spark_regression_evaluator_agrees(spark: SparkSession) -> None:
    frame = pd.DataFrame({"label": R_TRUE, "prediction": R_PRED})
    metrics = spark_regression_metrics(spark.createDataFrame(frame), label="label")
    expected = regression_metrics(R_TRUE, R_PRED)
    for name in ("rmse", "mae", "r2"):
        assert metrics[name] == pytest.approx(expected[name])


@pytest.mark.parametrize(
    "scenario,metric",
    [
        (Scenario.MISSED_POSITIVES_COSTLY, "recall"),
        (Scenario.FALSE_ALARMS_COSTLY, "precision"),
        (Scenario.IMBALANCED_BOTH_ERRORS_MATTER, "F1"),
        (Scenario.WELL_CALIBRATED_PROBABILITIES, "log loss"),
        (Scenario.RANKING_ACROSS_THRESHOLDS, "ROC AUC"),
        (Scenario.BALANCED_CLASSES_EQUAL_COSTS, "accuracy"),
        (Scenario.LARGE_ERRORS_ESPECIALLY_BAD, "RMSE"),
        (Scenario.ROBUST_TO_OUTLIERS, "MAE"),
        (Scenario.SHARE_OF_VARIANCE_EXPLAINED, "R-squared"),
    ],
)
def test_choose_metric(scenario: Scenario, metric: str) -> None:
    assert choose_metric(scenario) == metric
