"""Chapter 7 pitfalls: mean imputation of skewed data, leakage, high-cardinality one-hot, log of zero."""

import numpy as np
import pandas as pd
import pytest

from db_ml_associate.ch07_cleaning.pitfalls import (
    correct_impute_after_split,
    correct_index_high_cardinality,
    correct_log1p,
    correct_median_impute,
    wrong_impute_before_split,
    wrong_log_of_zero,
    wrong_mean_impute_skewed,
    wrong_one_hot_high_cardinality_for_trees,
)
from db_ml_associate.common import CHURN_LABEL, split_frame
from db_ml_associate.datasets import load_customer_churn


@pytest.fixture(scope="module")
def churn() -> pd.DataFrame:
    return load_customer_churn()


def test_mean_of_a_skewed_column_is_pulled_up_by_the_tail(churn: pd.DataFrame) -> None:
    mean_fill = wrong_mean_impute_skewed(churn, "data_usage_gb")
    median_fill = correct_median_impute(churn, "data_usage_gb")
    assert mean_fill - median_fill > 3
    typical = churn["data_usage_gb"].dropna()
    assert (typical < mean_fill).mean() > 0.6  # most real customers use less than the "typical" fill value


def test_imputing_before_the_split_uses_test_rows(churn: pd.DataFrame) -> None:
    train, _ = split_frame(churn, CHURN_LABEL)
    leaked = wrong_impute_before_split(churn, "monthly_charges")
    honest = correct_impute_after_split(churn, "monthly_charges")
    assert leaked == pytest.approx(churn["monthly_charges"].median())
    assert honest == pytest.approx(train["monthly_charges"].median())


def test_one_hot_encoding_a_high_cardinality_column_explodes_the_width(churn: pd.DataFrame) -> None:
    assert wrong_one_hot_high_cardinality_for_trees(churn, "city") > 700
    assert correct_index_high_cardinality(churn, "city") == 1


def test_log_of_zero_is_minus_infinity(churn: pd.DataFrame) -> None:
    with np.errstate(divide="ignore"):
        assert np.isneginf(wrong_log_of_zero(churn, "support_calls")).any()
    assert np.isfinite(correct_log1p(churn, "support_calls")).all()
