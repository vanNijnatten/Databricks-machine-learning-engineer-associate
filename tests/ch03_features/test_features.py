"""Customer feature computation (pandas and Spark) for the feature table (S1.06, S1.07)."""

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from db_ml_associate.ch03_features.features import (
    FEATURE_COLUMNS,
    compute_customer_features,
    compute_customer_features_pandas,
    label_frame,
)
from db_ml_associate.datasets import load_customer_churn


@pytest.fixture(scope="module")
def churn() -> pd.DataFrame:
    return load_customer_churn()


def test_one_row_per_customer_with_a_non_null_key(churn: pd.DataFrame) -> None:
    features = compute_customer_features_pandas(churn)
    assert features["customer_id"].is_unique
    assert features["customer_id"].notna().all()
    assert list(features.columns) == ["customer_id", *FEATURE_COLUMNS]
    assert "churned" not in features.columns  # the label is not a feature


def test_derived_features(churn: pd.DataFrame) -> None:
    features = compute_customer_features_pandas(churn).set_index("customer_id")
    row = churn.set_index("customer_id").loc[1]
    assert features.loc[1, "is_month_to_month"] == int(row["contract_type"] == "month-to-month")
    expected_rate = row["support_calls"] / max(row["tenure_months"] / 12, 1)
    assert features.loc[1, "support_calls_per_year"] == pytest.approx(expected_rate)


def test_label_frame_has_key_and_label_only(churn: pd.DataFrame) -> None:
    assert list(label_frame(churn).columns) == ["customer_id", "churned"]


@pytest.mark.spark
def test_spark_features_match_pandas(spark: SparkSession, churn: pd.DataFrame) -> None:
    spark_features = compute_customer_features(spark.createDataFrame(churn)).toPandas()
    pandas_features = compute_customer_features_pandas(churn)
    pd.testing.assert_frame_equal(
        spark_features.sort_values("customer_id").reset_index(drop=True),
        pandas_features.sort_values("customer_id").reset_index(drop=True),
        check_dtype=False,
    )
