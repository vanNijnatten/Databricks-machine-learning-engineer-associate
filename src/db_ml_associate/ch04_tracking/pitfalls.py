"""Chapter 4 pitfalls, each as a wrong and a correct version (constitution Principle V).

The ``wrong_*`` functions are deliberately incorrect — never copy them.
"""

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo

from db_ml_associate.ch09_pipelines.sklearn_pipeline import build_sklearn_churn_pipeline
from db_ml_associate.common import CHURN_LABEL


def _fit(train: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    features = train.drop(columns=CHURN_LABEL)
    return build_sklearn_churn_pipeline().fit(features, train[CHURN_LABEL]), features


def wrong_log_model_without_signature(train: pd.DataFrame) -> ModelInfo:
    """⚠️ WRONG: log a model without signature or input example.

    Unity Catalog refuses to register a model without a signature, and serving cannot validate input.
    """
    model, _ = _fit(train)
    with mlflow.start_run():
        return mlflow.sklearn.log_model(model, name="model")


def correct_log_model_with_signature(train: pd.DataFrame) -> ModelInfo:
    """Infer the signature from training data and predictions, and add an input example."""
    model, features = _fit(train)
    with mlflow.start_run():
        signature = infer_signature(features, model.predict(features))  # type: ignore[attr-defined]
        return mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=features.head(3))


def wrong_order_by_ascending_for_max_metric(client: MlflowClient, experiment_id: str, metric: str) -> Run:
    """⚠️ WRONG: ``order_by`` defaults to ascending — for a score this returns the *worst* run."""
    return client.search_runs([experiment_id], order_by=[f"metrics.{metric}"], max_results=1)[0]


def correct_order_by_descending_for_max_metric(client: MlflowClient, experiment_id: str, metric: str) -> Run:
    """Sort scores descending (and losses ascending) to get the best run first."""
    return client.search_runs([experiment_id], order_by=[f"metrics.{metric} DESC"], max_results=1)[0]
