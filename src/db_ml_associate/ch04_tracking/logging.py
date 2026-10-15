"""Logging parameters, metrics, artifacts and models to MLflow (exam objective S1.12).

On Databricks the tracking server is built in: runs land in the notebook's experiment unless you
call ``mlflow.set_experiment("/Users/<you>/churn")`` first.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, roc_auc_score

from db_ml_associate.ch09_pipelines.sklearn_pipeline import build_sklearn_churn_pipeline
from db_ml_associate.common import CHURN_CATEGORICAL, CHURN_FEATURES, CHURN_LABEL, CHURN_NUMERIC


@dataclass(frozen=True)
class LoggedRun:
    """Identifiers of a logged training run and its model."""

    run_id: str
    model_uri: str
    metrics: dict[str, float]


def train_and_log_churn_model(train: pd.DataFrame, test: pd.DataFrame, c: float = 1.0) -> LoggedRun:
    """Train the churn pipeline and log everything an auditor would want to see."""
    features, labels = train[list(CHURN_FEATURES)], train[CHURN_LABEL]  # the signature lists only real inputs
    test_features, test_labels = test[list(CHURN_FEATURES)], test[CHURN_LABEL]
    # --8<-- [start:manual_logging]
    with mlflow.start_run(run_name=f"logreg C={c}") as run:
        mlflow.log_params({"C": c, "class_weight": "balanced", "train_rows": len(train)})  # inputs
        model = build_sklearn_churn_pipeline().set_params(model__C=c).fit(features, labels)

        predictions = model.predict(test_features)
        metrics = {
            "test_f1": f1_score(test_labels, predictions),
            "test_roc_auc": roc_auc_score(test_labels, model.predict_proba(test_features)[:, 1]),
        }
        mlflow.log_metrics(metrics)  # results

        figure = ConfusionMatrixDisplay.from_predictions(test_labels, predictions).figure_
        mlflow.log_figure(figure, "confusion_matrix.png")  # any file: plots, reports, data samples
        mlflow.log_dict({"numeric": list(CHURN_NUMERIC), "categorical": list(CHURN_CATEGORICAL)}, "features.json")

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",  # MLflow 3; replaces the deprecated artifact_path
            signature=infer_signature(features, predictions),  # input/output schema
            input_example=features.head(3),
        )
    # --8<-- [end:manual_logging]
    plt.close(figure)
    return LoggedRun(run_id=run.info.run_id, model_uri=model_info.model_uri, metrics=metrics)


def log_learning_curve(losses: Sequence[float]) -> None:
    """Log one metric value per step (epoch, iteration) inside the active run."""
    # --8<-- [start:metric_steps]
    for step, loss in enumerate(losses):
        mlflow.log_metric("train_loss", loss, step=step)  # the UI draws a line chart over steps
    # --8<-- [end:metric_steps]


def train_with_autolog(train: pd.DataFrame) -> str:
    """Let MLflow autologging capture parameters, training metrics and the model."""
    features, labels = train.drop(columns=CHURN_LABEL), train[CHURN_LABEL]
    # --8<-- [start:autolog]
    mlflow.sklearn.autolog(log_models=True)  # or mlflow.autolog() for all supported libraries
    with mlflow.start_run() as run:
        build_sklearn_churn_pipeline().fit(features, labels)  # params, training_* metrics, model logged
    mlflow.sklearn.autolog(disable=True)
    # --8<-- [end:autolog]
    return run.info.run_id


def tune_with_nested_runs(train: pd.DataFrame, test: pd.DataFrame, cs: Sequence[float]) -> str:
    """One parent run for the tuning session, one child run per candidate."""
    features, labels = train.drop(columns=CHURN_LABEL), train[CHURN_LABEL]
    test_features, test_labels = test.drop(columns=CHURN_LABEL), test[CHURN_LABEL]
    # --8<-- [start:nested_runs]
    with mlflow.start_run(run_name="tune C") as parent:
        for c in cs:
            with mlflow.start_run(run_name=f"C={c}", nested=True):  # child run, linked to the parent
                model = build_sklearn_churn_pipeline().set_params(model__C=c).fit(features, labels)
                mlflow.log_param("C", c)
                mlflow.log_metric("test_f1", f1_score(test_labels, model.predict(test_features)))
    # --8<-- [end:nested_runs]
    return parent.info.run_id
