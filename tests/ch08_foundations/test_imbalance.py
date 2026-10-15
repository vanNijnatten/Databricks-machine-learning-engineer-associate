"""Class imbalance: weighting and resampling (S3.02)."""

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from db_ml_associate.ch08_foundations.imbalance import (
    add_spark_class_weights,
    balanced_class_weights,
    make_churn_classifier,
    minority_recall,
    random_oversample,
    random_undersample,
)
from db_ml_associate.common import CHURN_LABEL, split_frame
from db_ml_associate.datasets import load_customer_churn


@pytest.fixture(scope="module")
def churn_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    return split_frame(load_customer_churn(), CHURN_LABEL)


def test_balanced_weights_are_inverse_to_class_frequency() -> None:
    labels = pd.Series([0] * 90 + [1] * 10)
    weights = balanced_class_weights(labels)
    assert weights == pytest.approx({0: 100 / (2 * 90), 1: 100 / (2 * 10)})


def test_class_weighting_improves_minority_recall(
    churn_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = churn_split
    features, labels = train.drop(columns=CHURN_LABEL), train[CHURN_LABEL]
    unweighted = make_churn_classifier().fit(features, labels)
    weighted = make_churn_classifier(class_weight="balanced").fit(features, labels)
    test_features, test_labels = test.drop(columns=CHURN_LABEL), test[CHURN_LABEL]
    assert (
        minority_recall(weighted, test_features, test_labels)
        > minority_recall(unweighted, test_features, test_labels) + 0.2
    )


def test_oversampling_balances_only_the_given_split(
    churn_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = churn_split
    balanced = random_oversample(train, CHURN_LABEL, seed=1)
    counts = balanced[CHURN_LABEL].value_counts()
    assert counts[0] == counts[1] == (train[CHURN_LABEL] == 0).sum()
    assert test[CHURN_LABEL].mean() == pytest.approx(0.1, abs=0.01)  # untouched


def test_undersampling_keeps_all_minority_rows(
    churn_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, _ = churn_split
    balanced = random_undersample(train, CHURN_LABEL, seed=1)
    counts = balanced[CHURN_LABEL].value_counts()
    assert counts[0] == counts[1] == (train[CHURN_LABEL] == 1).sum()


def test_resampling_is_deterministic(
    churn_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, _ = churn_split
    pd.testing.assert_frame_equal(
        random_oversample(train, CHURN_LABEL, 3),
        random_oversample(train, CHURN_LABEL, 3),
    )


@pytest.mark.spark
def test_spark_weight_column(spark: SparkSession) -> None:
    frame = spark.createDataFrame(pd.DataFrame({"churned": [0] * 9 + [1]}))
    weighted = add_spark_class_weights(frame, "churned").toPandas()
    assert set(weighted["class_weight"].round(4)) == {round(10 / 18, 4), 5.0}
