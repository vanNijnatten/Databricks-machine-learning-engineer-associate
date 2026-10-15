"""Outliers, imputation, one-hot encoding and log transforms (S2.02, S2.05-S2.09)."""

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from scipy.stats import skew

from db_ml_associate.ch07_cleaning.encoding import one_hot_pandas, one_hot_spark
from db_ml_associate.ch07_cleaning.impute import (
    add_missing_indicator,
    impute_categorical_mode,
    impute_pandas,
    impute_spark,
)
from db_ml_associate.ch07_cleaning.outliers import iqr_bounds, remove_outliers_iqr, remove_outliers_stddev
from db_ml_associate.ch07_cleaning.transforms import log_transform_spark
from db_ml_associate.datasets import load_house_prices

NORMAL = [10.0 + (i % 11) for i in range(100)]  # values 10..20
OUTLIERS = [100.0, -50.0]


@pytest.fixture(scope="module")
def with_outliers(spark: SparkSession) -> DataFrame:
    rows = [(i, value) for i, value in enumerate(NORMAL + OUTLIERS)] + [(999, None)]
    return spark.createDataFrame(rows, "id INT, value DOUBLE")


@pytest.mark.spark
@pytest.mark.parametrize("remove", [remove_outliers_stddev, remove_outliers_iqr])
def test_outlier_filters_remove_exactly_the_injected_outliers(with_outliers: DataFrame, remove: object) -> None:
    kept = remove(with_outliers, "value")  # type: ignore[operator]
    values = {row["value"] for row in kept.collect()}
    assert not values & set(OUTLIERS)
    assert set(NORMAL) <= values
    assert kept.filter(F.col("value").isNull()).count() == 1  # missing values are kept for imputation


@pytest.mark.spark
def test_iqr_bounds(with_outliers: DataFrame) -> None:
    lower, upper = iqr_bounds(with_outliers, "value")
    q1, q3 = 12.0, 18.0
    assert (lower, upper) == pytest.approx((q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)), abs=1.0)


@pytest.fixture(scope="module")
def with_gaps(spark: SparkSession) -> DataFrame:
    rows = [(1.0, "a"), (2.0, "a"), (3.0, "b"), (10.0, "a"), (None, None), (2.0, "c")]
    return spark.createDataFrame(rows, "amount DOUBLE, category STRING")


@pytest.mark.spark
@pytest.mark.parametrize("strategy,expected", [("mean", 3.6), ("median", 2.0), ("mode", 2.0)])
def test_spark_imputer(with_gaps: DataFrame, strategy: str, expected: float) -> None:
    imputed = impute_spark(with_gaps, ["amount"], strategy).filter(F.col("category").isNull()).first()
    assert imputed is not None
    assert imputed["amount"] == pytest.approx(expected)


@pytest.mark.spark
def test_categorical_mode(with_gaps: DataFrame) -> None:
    filled = impute_categorical_mode(with_gaps, "category")
    assert filled.filter(F.col("category").isNull()).count() == 0
    assert filled.filter(F.col("amount").isNull()).first()["category"] == "a"  # type: ignore[index]


@pytest.mark.spark
def test_missing_indicator(with_gaps: DataFrame) -> None:
    flagged = add_missing_indicator(with_gaps, "amount")
    assert [row["amount_was_missing"] for row in flagged.collect()] == [0, 0, 0, 0, 1, 0]


@pytest.mark.parametrize("strategy,expected", [("mean", 3.6), ("median", 2.0), ("mode", 2.0)])
def test_pandas_imputation_matches_spark(strategy: str, expected: float) -> None:
    frame = pd.DataFrame({"amount": [1.0, 2.0, 3.0, 10.0, np.nan, 2.0]})
    assert impute_pandas(frame, "amount", strategy)["amount"].iloc[4] == pytest.approx(expected)


@pytest.mark.spark
@pytest.mark.parametrize("keep_unseen,size", [(False, 2), (True, 3)])
def test_one_hot_spark_vector_size(spark: SparkSession, keep_unseen: bool, size: int) -> None:
    frame = spark.createDataFrame([("x",), ("y",), ("z",), ("x",)], ["letter"])
    encoded = one_hot_spark(frame, ["letter"], keep_unseen=keep_unseen).first()
    assert encoded is not None
    # 3 categories (+1 reserved index for unseen values when kept) minus 1 dropped by dropLast
    assert encoded["letter_vector"].size == size


def test_one_hot_pandas_creates_a_column_per_category() -> None:
    frame = pd.DataFrame({"letter": ["x", "y", "z", "x"], "n": [1, 2, 3, 4]})
    encoded = one_hot_pandas(frame, ["letter"])
    assert sorted(column for column in encoded.columns if column.startswith("letter_")) == [
        "letter_x",
        "letter_y",
        "letter_z",
    ]


@pytest.mark.spark
def test_log_transform_reduces_skew(spark: SparkSession) -> None:
    houses = load_house_prices()
    transformed = log_transform_spark(spark.createDataFrame(houses), "price").toPandas()
    assert skew(houses["price"]) > 1
    assert abs(skew(transformed["price_log"])) < 0.5
    assert transformed["price_log"].to_numpy() == pytest.approx(np.log1p(houses["price"].to_numpy()))
