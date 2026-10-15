"""Chapter 9 pitfalls: preprocessing leakage and using an unfitted estimator."""

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from db_ml_associate.ch09_pipelines.pitfalls import (
    correct_fit_inside_pipeline,
    correct_fit_then_transform,
    wrong_fit_scaler_on_full_data,
    wrong_transform_with_unfitted_transformer,
)
from db_ml_associate.common import CHURN_LABEL, split_frame
from db_ml_associate.datasets import load_customer_churn


@pytest.fixture(scope="module")
def churn() -> pd.DataFrame:
    return load_customer_churn()


def test_scaler_fitted_on_all_data_learns_test_statistics(churn: pd.DataFrame) -> None:
    train, _ = split_frame(churn, CHURN_LABEL)
    leaked_mean, _, _ = wrong_fit_scaler_on_full_data(churn, "tenure_months")
    assert leaked_mean == pytest.approx(churn["tenure_months"].mean())
    assert leaked_mean != pytest.approx(train["tenure_months"].mean(), abs=1e-6)


def test_scaler_inside_pipeline_learns_training_statistics_only(
    churn: pd.DataFrame,
) -> None:
    train, _ = split_frame(churn, CHURN_LABEL)
    learned_mean, _ = correct_fit_inside_pipeline(churn, "tenure_months")
    assert learned_mean == pytest.approx(train["tenure_months"].mean())


def test_unfitted_transformer_raises(churn: pd.DataFrame) -> None:
    with pytest.raises(NotFittedError):
        wrong_transform_with_unfitted_transformer(churn, "tenure_months")


def test_fit_then_transform_works(churn: pd.DataFrame) -> None:
    scaled = correct_fit_then_transform(churn, "tenure_months")
    assert scaled.mean() == pytest.approx(0, abs=1e-9)
