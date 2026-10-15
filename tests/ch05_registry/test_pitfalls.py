"""Chapter 5 pitfalls: hard-coded version numbers and workspace-registry stages."""

import pandas as pd
import pytest
from mlflow import MlflowClient

from db_ml_associate.ch04_tracking.logging import train_and_log_churn_model
from db_ml_associate.ch05_registry.pitfalls import (
    correct_load_by_alias,
    wrong_load_by_version_number,
    wrong_transition_stage,
)
from db_ml_associate.ch05_registry.registry import register_model, set_alias
from db_ml_associate.common import CHURN_LABEL, split_frame
from db_ml_associate.datasets import load_customer_churn

NAME = "ml_associate.tutorial.churn_model"


@pytest.fixture(scope="module")
def churn_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = split_frame(load_customer_churn(), CHURN_LABEL)
    return train.sample(n=800, random_state=0), test


def test_hard_coded_version_misses_the_promotion(
    local_mlflow: str, churn_split: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, test = churn_split
    for c in (0.01, 1.0):
        register_model(train_and_log_churn_model(train, test, c=c).model_uri, NAME)
    client = MlflowClient()
    set_alias(client, NAME, "champion", "1")
    job_uri = "models:/ml_associate.tutorial.churn_model/1"  # written into the job when v1 was champion
    set_alias(client, NAME, "champion", "2")  # promotion
    assert wrong_load_by_version_number(job_uri) == "1"  # still the old model
    assert correct_load_by_alias(NAME, "champion") == "2"  # follows the promotion


def test_stages_are_the_legacy_workflow() -> None:
    assert "not supported" in (wrong_transition_stage.__doc__ or "")
