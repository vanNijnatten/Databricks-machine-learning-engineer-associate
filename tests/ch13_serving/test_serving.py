"""Custom pyfunc models, endpoint configuration, traffic splits and queries (S4.02, S4.05, S4.06)."""

from unittest.mock import MagicMock

import mlflow
import pandas as pd
import pytest
from databricks.sdk.service.serving import EndpointCoreConfigInput

from db_ml_associate.ch09_pipelines.sklearn_pipeline import build_sklearn_churn_pipeline
from db_ml_associate.ch13_serving.custom_model import log_custom_model
from db_ml_associate.ch13_serving.workspace import (
    ServedVersion,
    create_endpoint,
    dataframe_split_payload,
    delete_endpoint,
    endpoint_config,
    query_endpoint,
    update_traffic,
)
from db_ml_associate.common import CHURN_CATEGORICAL, CHURN_LABEL, CHURN_NUMERIC, split_frame
from db_ml_associate.datasets import load_customer_churn

FEATURES = [*CHURN_NUMERIC, *CHURN_CATEGORICAL]
MODEL = "ml_associate.tutorial.churn_model"
CHAMPION = ServedVersion(name="champion", model_name=MODEL, version="1")
CHALLENGER = ServedVersion(name="challenger", model_name=MODEL, version="2")


@pytest.fixture(scope="module")
def churn_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = split_frame(load_customer_churn(), CHURN_LABEL)
    return train.head(1500), test.head(10)


def test_custom_pyfunc_model_adds_business_logic(
    local_mlflow: str, churn_split: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, test = churn_split
    classifier = build_sklearn_churn_pipeline().fit(train[FEATURES], train[CHURN_LABEL])
    info = log_custom_model(classifier, train[FEATURES].head(5))
    assert info.signature is not None
    predictions = mlflow.pyfunc.load_model(info.model_uri).predict(test[FEATURES])
    assert list(predictions.columns) == ["churn_probability", "risk_band"]
    assert set(predictions["risk_band"]) <= {"low", "medium", "high"}
    assert predictions["churn_probability"].between(0, 1).all()


def test_endpoint_config_with_a_traffic_split() -> None:
    config = endpoint_config([CHAMPION, CHALLENGER], {"champion": 80, "challenger": 20})
    assert isinstance(config, EndpointCoreConfigInput)
    entities = config.served_entities or []
    assert [(e.name, e.entity_name, e.entity_version) for e in entities] == [
        ("champion", MODEL, "1"),
        ("challenger", MODEL, "2"),
    ]
    assert all(e.workload_size == "Small" and e.scale_to_zero_enabled for e in entities)
    assert config.traffic_config is not None
    routes = config.traffic_config.routes or []
    assert {(r.served_model_name, r.traffic_percentage) for r in routes} == {("champion", 80), ("challenger", 20)}


@pytest.mark.parametrize(
    "traffic",
    [{"champion": 80, "challenger": 30}, {"champion": 100}, {"champion": 50, "unknown": 50}],
)
def test_invalid_traffic_splits_are_rejected(traffic: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="traffic"):
        endpoint_config([CHAMPION, CHALLENGER], traffic)


def test_create_update_and_delete_use_the_sdk() -> None:
    workspace = MagicMock()
    config = endpoint_config([CHAMPION], {"champion": 100})
    create_endpoint(workspace, "churn-endpoint", config)
    workspace.serving_endpoints.create_and_wait.assert_called_once_with(name="churn-endpoint", config=config)
    update_traffic(workspace, "churn-endpoint", [CHAMPION, CHALLENGER], {"champion": 90, "challenger": 10})
    kwargs = workspace.serving_endpoints.update_config_and_wait.call_args.kwargs
    assert kwargs["name"] == "churn-endpoint"
    assert len(kwargs["served_entities"]) == 2
    delete_endpoint(workspace, "churn-endpoint")
    workspace.serving_endpoints.delete.assert_called_once_with(name="churn-endpoint")


def test_dataframe_split_payload() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    payload = dataframe_split_payload(frame)
    assert payload == {"dataframe_split": {"columns": ["a", "b"], "data": [[1.0, "x"], [2.0, "y"]]}}


def test_query_through_the_mlflow_deployments_client() -> None:
    client = MagicMock()
    client.predict.return_value = {"predictions": [0, 1]}
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    assert query_endpoint(client, "churn-endpoint", frame) == [0, 1]
    client.predict.assert_called_once_with(endpoint="churn-endpoint", inputs=dataframe_split_payload(frame))
