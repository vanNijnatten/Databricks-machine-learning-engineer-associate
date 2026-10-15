"""Chapter 11 pitfall: metrics on a log-transformed target must be computed after exponentiating (S3.14)."""

import numpy as np
import pandas as pd
import pytest

from db_ml_associate.ch11_evaluation.pitfalls import (
    correct_exponentiate_before_metrics,
    wrong_rmse_on_log_scale,
)
from db_ml_associate.datasets import load_house_prices


@pytest.fixture(scope="module")
def houses() -> pd.DataFrame:
    return load_house_prices()


def test_rmse_on_log_scale_is_not_in_price_units(houses: pd.DataFrame) -> None:
    wrong = wrong_rmse_on_log_scale(houses)
    correct = correct_exponentiate_before_metrics(houses)
    assert wrong < 1  # log units: looks tiny and is not comparable to prices
    assert 10_000 < correct.rmse < 100_000  # euros
    assert correct.predictions.min() > 1_000  # predictions are prices, not logs


def test_back_transform_inverts_log1p() -> None:
    prices = np.array([100_000.0, 250_000.0])
    assert np.expm1(np.log1p(prices)) == pytest.approx(prices)
