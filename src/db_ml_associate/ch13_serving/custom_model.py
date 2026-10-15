"""A custom pyfunc model: a classifier plus business post-processing (exam objective S4.02).

A *custom model* is any Python logic wrapped in ``mlflow.pyfunc.PythonModel``. Model Serving
deploys it like any other MLflow model, so preprocessing, post-processing or several models can run
behind one endpoint.
"""

import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo


# --8<-- [start:python_model]
class ChurnRiskModel(mlflow.pyfunc.PythonModel):
    """Return the churn probability and a risk band the retention team can act on."""

    def load_context(self, context: Any) -> None:
        # Runs once when the endpoint (or load_model) starts: load artifacts here, not in predict.
        self.classifier = mlflow.sklearn.load_model(context.artifacts["classifier"])

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        probability = self.classifier.predict_proba(model_input)[:, 1]
        risk_band = np.select([probability >= 0.7, probability >= 0.4], ["high", "medium"], default="low")
        return pd.DataFrame({"churn_probability": probability, "risk_band": risk_band})


# --8<-- [end:python_model]


def log_custom_model(classifier: Any, input_example: pd.DataFrame) -> ModelInfo:
    """Save the fitted classifier as an artifact and log the wrapper as a pyfunc model."""
    with tempfile.TemporaryDirectory() as directory:
        classifier_path = Path(directory) / "classifier"
        mlflow.sklearn.save_model(classifier, classifier_path)
        wrapper = ChurnRiskModel()
        wrapper.classifier = classifier  # only to compute an example output for the signature
        example_output = wrapper.predict(None, input_example)
        # --8<-- [start:log_pyfunc]
        with mlflow.start_run():
            model_info = mlflow.pyfunc.log_model(
                name="churn_risk_model",
                python_model=ChurnRiskModel(),
                artifacts={"classifier": str(classifier_path)},  # copied into the logged model
                signature=infer_signature(input_example, example_output),  # required for Unity Catalog
                input_example=input_example,
            )
        # --8<-- [end:log_pyfunc]
    return model_info
