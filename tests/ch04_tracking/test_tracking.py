"""Manual logging, autologging, nested runs and finding the best run (S1.11, S1.12)."""

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from db_ml_associate.ch04_tracking.best_run import find_best_run, runs_as_dataframe
from db_ml_associate.ch04_tracking.logging import (
    log_learning_curve,
    train_and_log_churn_model,
    train_with_autolog,
    tune_with_nested_runs,
)
from db_ml_associate.common import CHURN_LABEL, split_frame
from db_ml_associate.datasets import load_customer_churn


@pytest.fixture(scope="module")
def churn_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = split_frame(load_customer_churn(), CHURN_LABEL)
    return train.sample(n=1500, random_state=0), test


def test_manual_logging_records_params_metrics_artifacts_and_model(
    local_mlflow: str, churn_split: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, test = churn_split
    logged = train_and_log_churn_model(train, test, c=1.0)
    run = MlflowClient().get_run(logged.run_id)
    assert run.data.params["C"] == "1.0"
    assert {"test_f1", "test_roc_auc"} <= set(run.data.metrics)
    artifacts = {item.path for item in MlflowClient().list_artifacts(logged.run_id)}
    assert {"confusion_matrix.png", "features.json"} <= artifacts
    model = mlflow.models.get_model_info(logged.model_uri)
    assert model.signature is not None
    loaded = mlflow.pyfunc.load_model(logged.model_uri)
    assert len(loaded.predict(test.drop(columns=CHURN_LABEL).head(5))) == 5


def test_metric_history_with_steps(local_mlflow: str) -> None:
    with mlflow.start_run() as run:
        log_learning_curve([0.5, 0.4, 0.35])
    history = MlflowClient().get_metric_history(run.info.run_id, "train_loss")
    assert [(point.step, point.value) for point in history] == [(0, 0.5), (1, 0.4), (2, 0.35)]


def test_autolog_logs_params_and_metrics(local_mlflow: str, churn_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, _ = churn_split
    run_id = train_with_autolog(train)
    run = MlflowClient().get_run(run_id)
    assert "model__C" in run.data.params or "C" in run.data.params
    assert any(name.startswith("training_") for name in run.data.metrics)


def test_nested_runs_and_finding_the_best(local_mlflow: str, churn_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = churn_split
    parent_id = tune_with_nested_runs(train, test, cs=(0.01, 1.0, 100.0))
    client = MlflowClient()
    experiment = client.get_experiment_by_name(local_mlflow)
    assert experiment is not None
    children = client.search_runs([experiment.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{parent_id}'")
    assert len(children) == 3
    best = find_best_run(client, experiment.experiment_id, "test_f1")
    assert best.data.metrics["test_f1"] == max(child.data.metrics["test_f1"] for child in children)


def test_find_best_run_for_a_metric_to_minimise(local_mlflow: str) -> None:
    for loss in (0.3, 0.1, 0.2):
        with mlflow.start_run():
            mlflow.log_metric("log_loss", loss)
    experiment = mlflow.get_experiment_by_name(local_mlflow)
    assert experiment is not None
    best = find_best_run(MlflowClient(), experiment.experiment_id, "log_loss", mode="min")
    assert best.data.metrics["log_loss"] == 0.1


def test_runs_as_dataframe(local_mlflow: str) -> None:
    for value in (1, 2):
        with mlflow.start_run():
            mlflow.log_metric("score", value)
    frame = runs_as_dataframe(local_mlflow)
    assert sorted(frame["metrics.score"]) == [1, 2]
