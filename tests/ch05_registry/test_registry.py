"""Registering models, tags and champion/challenger aliases (S1.14, S1.17, S1.18).

Uses a throw-away local SQLite registry, which supports aliases and tags like Unity Catalog does.
"""

from unittest.mock import MagicMock

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from db_ml_associate.ch04_tracking.logging import train_and_log_churn_model
from db_ml_associate.ch05_registry.registry import (
    copy_to_environment,
    load_by_alias,
    promote_challenger,
    register_model,
    remove_model_tag,
    remove_version_tag,
    set_alias,
    set_model_tag,
    set_version_tag,
    validate_uc_name,
)
from db_ml_associate.common import CHURN_LABEL, split_frame
from db_ml_associate.datasets import load_customer_churn

NAME = "ml_associate.tutorial.churn_model"


@pytest.fixture(scope="module")
def churn_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = split_frame(load_customer_churn(), CHURN_LABEL)
    return train.sample(n=1200, random_state=0), test


@pytest.mark.parametrize("name", ["churn_model", "tutorial.churn_model", "a.b.c.d"])
def test_unity_catalog_names_have_three_levels(name: str) -> None:
    with pytest.raises(ValueError, match="catalog.schema.model"):
        validate_uc_name(name)


def test_register_two_versions(local_mlflow: str, churn_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = churn_split
    first = register_model(train_and_log_churn_model(train, test, c=0.01).model_uri, NAME)
    second = register_model(train_and_log_churn_model(train, test, c=1.0).model_uri, NAME)
    assert (str(first.version), str(second.version)) == ("1", "2")


def test_register_from_run_uri(local_mlflow: str, churn_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = churn_split
    logged = train_and_log_churn_model(train, test)
    version = register_model(f"runs:/{logged.run_id}/model", NAME)
    assert version.run_id == logged.run_id


def test_tags_on_model_and_version(local_mlflow: str, churn_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = churn_split
    register_model(train_and_log_churn_model(train, test).model_uri, NAME)
    client = MlflowClient()
    set_model_tag(client, NAME, "team", "retention")
    set_version_tag(client, NAME, "1", "validation_status", "approved")
    assert client.get_registered_model(NAME).tags["team"] == "retention"
    assert client.get_model_version(NAME, "1").tags["validation_status"] == "approved"
    remove_model_tag(client, NAME, "team")
    remove_version_tag(client, NAME, "1", "validation_status")
    assert "team" not in client.get_registered_model(NAME).tags
    assert "validation_status" not in client.get_model_version(NAME, "1").tags


def test_champion_challenger_promotion(local_mlflow: str, churn_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = churn_split
    weak = train_and_log_churn_model(train, test, c=0.0001)  # heavily regularised: worse model
    strong = train_and_log_churn_model(train, test, c=1.0)
    register_model(weak.model_uri, NAME)
    register_model(strong.model_uri, NAME)
    client = MlflowClient()
    set_alias(client, NAME, "champion", "1")
    set_alias(client, NAME, "challenger", "2")

    assert promote_challenger(client, NAME, "test_f1") is True
    assert str(client.get_model_version_by_alias(NAME, "champion").version) == "2"
    with pytest.raises(mlflow.exceptions.MlflowException):
        client.get_model_version_by_alias(NAME, "challenger")  # removed after promotion
    assert len(load_by_alias(NAME, "champion").predict(test.drop(columns=CHURN_LABEL).head(3))) == 3


def test_worse_challenger_is_not_promoted(local_mlflow: str, churn_split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, test = churn_split
    register_model(train_and_log_churn_model(train, test, c=1.0).model_uri, NAME)
    register_model(train_and_log_churn_model(train, test, c=0.0001).model_uri, NAME)
    client = MlflowClient()
    set_alias(client, NAME, "champion", "1")
    set_alias(client, NAME, "challenger", "2")
    assert promote_challenger(client, NAME, "test_f1") is False
    assert str(client.get_model_version_by_alias(NAME, "champion").version) == "1"


def test_copy_to_environment_calls_copy_model_version() -> None:
    client = MagicMock()
    copy_to_environment(client, "staging.ml.churn_model", "3", "prod.ml.churn_model")
    client.copy_model_version.assert_called_once_with(
        src_model_uri="models:/staging.ml.churn_model/3", dst_name="prod.ml.churn_model"
    )
