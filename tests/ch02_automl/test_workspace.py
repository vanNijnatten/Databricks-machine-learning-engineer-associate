"""AutoML wrappers call the databricks.automl API correctly (S1.03, S1.04)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from db_ml_associate.ch02_automl.workspace import (
    import_trial_notebook,
    run_automl_classification,
    run_automl_regression,
    summarize_automl,
)


def fake_summary() -> SimpleNamespace:
    best = SimpleNamespace(
        mlflow_run_id="run-123",
        metrics={"val_f1_score": 0.41},
        notebook_url="https://example/notebook",
        model_path="dbfs:/databricks/mlflow/run-123/artifacts/model",
        model_description="LGBMClassifier(max_depth=6)",
        artifact_uri="dbfs:/.../notebooks/trial.ipynb",
    )
    return SimpleNamespace(
        experiment=SimpleNamespace(experiment_id="42"),
        best_trial=best,
        trials=[best, SimpleNamespace(mlflow_run_id="run-456")],
    )


def test_classification_call() -> None:
    automl = MagicMock()
    frame = pd.DataFrame({"churned": [0, 1]})
    run_automl_classification(automl, frame, "churned", timeout_minutes=30, exclude_cols=["customer_id"])
    automl.classify.assert_called_once_with(
        dataset=frame,
        target_col="churned",
        primary_metric="f1",
        timeout_minutes=30,
        exclude_cols=["customer_id"],
    )


def test_regression_uses_regress_and_r2_default() -> None:
    automl = MagicMock()
    run_automl_regression(automl, "ml_associate.tutorial.house_prices", "price", timeout_minutes=15)
    kwargs = automl.regress.call_args.kwargs
    assert kwargs["primary_metric"] == "r2"
    assert kwargs["dataset"] == "ml_associate.tutorial.house_prices"


def test_timeout_has_a_minimum_of_five_minutes() -> None:
    with pytest.raises(ValueError, match="5"):
        run_automl_classification(MagicMock(), pd.DataFrame(), "churned", timeout_minutes=2)


def test_summarize() -> None:
    result = summarize_automl(fake_summary())  # type: ignore[arg-type]
    assert result.experiment_id == "42"
    assert result.best_run_id == "run-123"
    assert result.best_metrics == {"val_f1_score": 0.41}
    assert result.best_model_uri == "runs:/run-123/model"
    assert result.trial_count == 2


def test_import_trial_notebook() -> None:
    automl = MagicMock()
    automl.import_notebook.return_value = SimpleNamespace(url="https://example/imported", path="/Users/me/trial")
    url = import_trial_notebook(automl, fake_summary().best_trial, "/Users/me/trial")  # type: ignore[arg-type]
    automl.import_notebook.assert_called_once_with("dbfs:/.../notebooks/trial.ipynb", "/Users/me/trial", overwrite=True)
    assert url == "https://example/imported"


@pytest.mark.databricks
def test_real_automl_run() -> None:  # pragma: no cover - runs only inside a Databricks workspace
    from databricks import automl  # type: ignore[import-not-found]

    from db_ml_associate.datasets import load_customer_churn

    summary = run_automl_classification(automl, load_customer_churn(), "churned", timeout_minutes=5)
    assert summarize_automl(summary).trial_count >= 1
