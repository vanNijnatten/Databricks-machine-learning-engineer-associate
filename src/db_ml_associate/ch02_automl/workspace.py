"""Running Databricks AutoML from Python (exam objectives S1.03, S1.04).

Requires a Databricks workspace. The ``automl`` module is passed in (``from databricks import
automl``) so that the code can be tested without a workspace.

Platform change: AutoML is not a built-in library from Databricks Runtime 18.0 ML; install the
``databricks-automl-runtime`` package as described in the AutoML documentation, and run it on
**dedicated (classic) compute** — AutoML does not support standard access mode.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

MINIMUM_TIMEOUT_MINUTES = 5


@dataclass(frozen=True)
class AutoMLResult:
    """The parts of an ``AutoMLSummary`` a learner needs next."""

    experiment_id: str
    best_run_id: str
    best_metrics: dict[str, float]
    best_model_uri: str
    best_notebook_url: str | None
    trial_count: int


def _check_timeout(timeout_minutes: int) -> None:
    if timeout_minutes < MINIMUM_TIMEOUT_MINUTES:
        raise ValueError(f"AutoML timeout_minutes must be at least {MINIMUM_TIMEOUT_MINUTES}")


def run_automl_classification(
    automl: Any,
    dataset: Any,
    target: str,
    timeout_minutes: int = 30,
    primary_metric: str = "f1",
    exclude_cols: Sequence[str] = (),
) -> Any:
    """Start an AutoML classification experiment and return its ``AutoMLSummary``."""
    _check_timeout(timeout_minutes)
    # --8<-- [start:classify]
    summary = automl.classify(
        dataset=dataset,  # Spark/pandas DataFrame or a Unity Catalog table name
        target_col=target,
        primary_metric=primary_metric,  # f1 (default), log_loss, precision, accuracy, roc_auc
        timeout_minutes=timeout_minutes,  # default 120, minimum 5
        exclude_cols=list(exclude_cols),  # e.g. identifiers that must not be features
    )
    # --8<-- [end:classify]
    return summary


def run_automl_regression(
    automl: Any,
    dataset: Any,
    target: str,
    timeout_minutes: int = 30,
    primary_metric: str = "r2",
    exclude_cols: Sequence[str] = (),
) -> Any:
    """Start an AutoML regression experiment (primary metrics: r2, mae, rmse, mse)."""
    _check_timeout(timeout_minutes)
    return automl.regress(
        dataset=dataset,
        target_col=target,
        primary_metric=primary_metric,
        timeout_minutes=timeout_minutes,
        exclude_cols=list(exclude_cols),
    )


def summarize_automl(summary: Any) -> AutoMLResult:
    """Extract the experiment, the best trial's run, metrics and model URI."""
    # --8<-- [start:summary]
    best = summary.best_trial  # TrialInfo with the best primary-metric score on validation data
    result = AutoMLResult(
        experiment_id=summary.experiment.experiment_id,  # every trial is an MLflow run here
        best_run_id=best.mlflow_run_id,
        best_metrics=dict(best.metrics),
        best_model_uri=f"runs:/{best.mlflow_run_id}/model",  # or best.load_model()
        best_notebook_url=best.notebook_url,  # generated, editable source code
        trial_count=len(summary.trials),
    )
    # --8<-- [end:summary]
    return result


def import_trial_notebook(automl: Any, trial: Any, path: str) -> str:
    """Import a (non-best) trial's notebook from its MLflow artifact into the workspace."""
    # --8<-- [start:import_notebook]
    result = automl.import_notebook(trial.artifact_uri, path, overwrite=True)
    # --8<-- [end:import_notebook]
    return str(result.url)
