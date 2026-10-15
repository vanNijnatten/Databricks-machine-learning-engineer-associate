"""Chapter 10 pitfalls: tuning on the test set and maximising with a minimiser."""

import pandas as pd
import pytest

from db_ml_associate.ch10_tuning.pitfalls import (
    TunedScore,
    correct_return_negative_metric,
    correct_tune_with_cv_then_test,
    wrong_fmin_maximise_metric,
    wrong_tune_on_test_set,
)
from db_ml_associate.datasets import load_customer_churn


@pytest.fixture(scope="module")
def churn() -> pd.DataFrame:
    return load_customer_churn()


@pytest.fixture(scope="module")
def wrong(churn: pd.DataFrame) -> TunedScore:
    return wrong_tune_on_test_set(churn)


@pytest.mark.slow
def test_score_of_a_model_chosen_on_the_test_set_is_optimistic(
    wrong: TunedScore,
) -> None:
    assert wrong.reported - wrong.on_new_data > 0.02


@pytest.mark.slow
def test_cv_tuning_reports_an_honest_estimate(churn: pd.DataFrame, wrong: TunedScore) -> None:
    correct = correct_tune_with_cv_then_test(churn)
    assert abs(correct.reported - correct.on_new_data) < wrong.reported - wrong.on_new_data


def test_returning_the_metric_itself_makes_fmin_pick_the_worst_model(
    churn: pd.DataFrame,
) -> None:
    worst = wrong_fmin_maximise_metric(churn)
    best = correct_return_negative_metric(churn)
    assert worst.f1 < best.f1 - 0.1
