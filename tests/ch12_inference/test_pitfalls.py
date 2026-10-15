"""Chapter 12 pitfalls: loading the model per row, and columns that do not match the signature."""

import mlflow
import pandas as pd
import pytest
from mlflow.models import infer_signature

from db_ml_associate.ch09_pipelines.sklearn_pipeline import build_sklearn_churn_pipeline
from db_ml_associate.ch12_inference.pitfalls import (
    correct_load_once,
    correct_matching_columns,
    wrong_load_model_per_row,
    wrong_renamed_columns,
)
from db_ml_associate.common import CHURN_CATEGORICAL, CHURN_LABEL, CHURN_NUMERIC, split_frame
from db_ml_associate.datasets import load_customer_churn

FEATURES = [*CHURN_NUMERIC, *CHURN_CATEGORICAL]


@pytest.fixture
def model_uri(local_mlflow: str) -> str:
    train, _ = split_frame(load_customer_churn(), CHURN_LABEL)
    features = train[FEATURES].head(1000)
    model = build_sklearn_churn_pipeline().fit(features, train[CHURN_LABEL].head(1000))
    with mlflow.start_run():
        return mlflow.sklearn.log_model(
            model, name="model", signature=infer_signature(features, model.predict(features))
        ).model_uri


@pytest.fixture(scope="module")
def batch() -> pd.DataFrame:
    return load_customer_churn()[FEATURES].head(20)


def test_loading_per_row_loads_the_model_once_per_row(model_uri: str, batch: pd.DataFrame) -> None:
    wrong = wrong_load_model_per_row(model_uri, batch)
    correct = correct_load_once(model_uri, batch)
    assert wrong.loads == len(batch)
    assert correct.loads == 1
    assert wrong.predictions == correct.predictions


def test_renamed_columns_fail_schema_enforcement(model_uri: str, batch: pd.DataFrame) -> None:
    with pytest.raises(mlflow.exceptions.MlflowException):
        wrong_renamed_columns(model_uri, batch)
    assert len(correct_matching_columns(model_uri, batch)) == len(batch)
