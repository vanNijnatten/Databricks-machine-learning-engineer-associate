"""The synthetic tutorial datasets are deterministic and contain the phenomena the book teaches."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skew

from db_ml_associate.datasets import (
    load_customer_churn,
    load_house_prices,
    make_customer_churn,
    make_house_prices,
)


@pytest.fixture(scope="module")
def churn() -> pd.DataFrame:
    return make_customer_churn(n_rows=5000, seed=42)


@pytest.fixture(scope="module")
def houses() -> pd.DataFrame:
    return make_house_prices(seed=42)


def test_churn_is_deterministic(churn: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(churn, make_customer_churn(n_rows=5000, seed=42))


def test_churn_seed_changes_data(churn: pd.DataFrame) -> None:
    assert not churn.equals(make_customer_churn(n_rows=5000, seed=7))


def test_churn_primary_key_is_unique(churn: pd.DataFrame) -> None:
    assert churn["customer_id"].is_unique
    assert churn["customer_id"].notna().all()


def test_churn_is_imbalanced(churn: pd.DataFrame) -> None:
    positive_rate = churn["churned"].mean()
    assert 0.09 <= positive_rate <= 0.11


def test_churn_has_a_numeric_column_with_about_five_percent_nulls(
    churn: pd.DataFrame,
) -> None:
    numeric = churn.select_dtypes("number")
    null_rates = numeric.isna().mean()
    assert ((null_rates > 0.04) & (null_rates < 0.06)).any()


def test_churn_has_low_and_high_cardinality_categoricals(churn: pd.DataFrame) -> None:
    categorical = churn.select_dtypes("object")
    cardinalities = categorical.nunique()
    assert (cardinalities <= 5).any()
    assert (cardinalities >= 500).any()


def test_churn_has_outliers_beyond_three_standard_deviations(
    churn: pd.DataFrame,
) -> None:
    calls = churn["support_calls"]
    z_scores = (calls - calls.mean()) / calls.std()
    assert (z_scores.abs() > 3).any()


def test_house_prices_target_is_right_skewed(houses: pd.DataFrame) -> None:
    assert skew(houses["price"]) > 1
    assert abs(skew(np.log1p(houses["price"]))) < 0.5


def test_house_prices_is_deterministic(houses: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(houses, make_house_prices(seed=42))


def test_committed_churn_file_matches_generator() -> None:
    pd.testing.assert_frame_equal(load_customer_churn(), make_customer_churn())


def test_committed_house_prices_file_matches_generator() -> None:
    pd.testing.assert_frame_equal(load_house_prices(), make_house_prices())
