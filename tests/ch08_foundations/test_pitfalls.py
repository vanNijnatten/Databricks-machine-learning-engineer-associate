"""Chapter 8 pitfalls show the stated consequence, and the correct version avoids it."""

import pandas as pd
import pytest

from db_ml_associate.ch08_foundations.pitfalls import (
    correct_resample_after_split,
    correct_use_recall_f1,
    wrong_accuracy_on_imbalanced,
    wrong_resample_before_split,
)
from db_ml_associate.datasets import load_customer_churn


@pytest.fixture(scope="module")
def churn() -> pd.DataFrame:
    return load_customer_churn()


def test_resampling_before_the_split_inflates_the_score(churn: pd.DataFrame) -> None:
    leaked = wrong_resample_before_split(churn)
    honest = correct_resample_after_split(churn)
    assert leaked > honest + 0.3


def test_accuracy_hides_a_useless_model(churn: pd.DataFrame) -> None:
    wrong = wrong_accuracy_on_imbalanced(churn)
    assert wrong["accuracy"] == pytest.approx(0.9, abs=0.01)
    assert wrong["recall"] == 0.0


def test_recall_and_f1_reveal_minority_performance(churn: pd.DataFrame) -> None:
    correct = correct_use_recall_f1(churn)
    assert correct["recall"] > 0.5
    assert correct["f1"] > 0.25
    assert correct["accuracy"] < wrong_accuracy_on_imbalanced(churn)["accuracy"]
