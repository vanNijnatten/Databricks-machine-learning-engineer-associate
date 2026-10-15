"""Chapter 6 pitfall: missing values that arrive in Spark as NaN instead of null."""

import math

import pytest
from pyspark.sql import SparkSession

from db_ml_associate.ch06_exploring.pitfalls import (
    correct_summary_after_nan_to_null,
    nan_to_null,
    wrong_summary_with_nan,
)
from db_ml_associate.datasets import load_customer_churn


@pytest.mark.spark
def test_nan_breaks_summary_statistics(spark: SparkSession) -> None:
    frame = spark.createDataFrame(load_customer_churn())
    wrong = wrong_summary_with_nan(frame, "data_usage_gb")
    assert wrong["count"] == "5000"  # NaN rows are counted as values
    assert math.isnan(float(wrong["mean"]))
    assert math.isnan(float(wrong["max"]))  # NaN sorts above every number


@pytest.mark.spark
def test_nan_to_null_restores_correct_statistics(spark: SparkSession) -> None:
    frame = spark.createDataFrame(load_customer_churn())
    correct = correct_summary_after_nan_to_null(frame, "data_usage_gb")
    assert correct["count"] == "4800"  # 4 % missing excluded
    assert 15 < float(correct["mean"]) < 25
    assert not math.isnan(float(correct["max"]))


@pytest.mark.spark
def test_nan_to_null_only_touches_nan(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(1.0,), (float("nan"),), (None,)], ["x"])
    values = [row["x"] for row in nan_to_null(frame, ["x"]).collect()]
    assert values == [1.0, None, None]
