"""Chapter 4 pitfalls: models without a signature and sorting the wrong way."""

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from db_ml_associate.ch04_tracking.pitfalls import (
    correct_log_model_with_signature,
    correct_order_by_descending_for_max_metric,
    wrong_log_model_without_signature,
    wrong_order_by_ascending_for_max_metric,
)
from db_ml_associate.common import CHURN_LABEL, split_frame
from db_ml_associate.datasets import load_customer_churn


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    train, _ = split_frame(load_customer_churn(), CHURN_LABEL)
    return train.sample(n=800, random_state=0)


def test_model_logged_without_signature_has_none(local_mlflow: str, train: pd.DataFrame) -> None:
    assert wrong_log_model_without_signature(train).signature is None
    assert correct_log_model_with_signature(train).signature is not None


def test_ascending_order_returns_the_worst_run(local_mlflow: str) -> None:
    for f1 in (0.2, 0.5, 0.4):
        with mlflow.start_run():
            mlflow.log_metric("f1", f1)
    experiment = mlflow.get_experiment_by_name(local_mlflow)
    assert experiment is not None
    client = MlflowClient()
    assert wrong_order_by_ascending_for_max_metric(client, experiment.experiment_id, "f1").data.metrics["f1"] == 0.2
    assert correct_order_by_descending_for_max_metric(client, experiment.experiment_id, "f1").data.metrics["f1"] == 0.5
