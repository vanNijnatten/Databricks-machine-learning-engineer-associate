"""Chapter 12 pitfalls (constitution Principle V).

The ``wrong_*`` functions are deliberately incorrect — never copy them.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlflow
import pandas as pd


@dataclass(frozen=True)
class ScoringRun:
    """Predictions and how many times the model was loaded to produce them."""

    predictions: list[float]
    loads: int


def _counting_loader(model_uri: str) -> tuple[Callable[[], Any], list[int]]:
    calls = [0]

    def load() -> Any:
        calls[0] += 1
        return mlflow.pyfunc.load_model(model_uri)

    return load, calls


def wrong_load_model_per_row(model_uri: str, frame: pd.DataFrame) -> ScoringRun:
    """⚠️ WRONG: load the model inside the row loop — one (slow) load per row."""
    load, calls = _counting_loader(model_uri)
    predictions = [float(load().predict(frame.iloc[[index]])[0]) for index in range(len(frame))]
    return ScoringRun(predictions=predictions, loads=calls[0])


def correct_load_once(model_uri: str, frame: pd.DataFrame) -> ScoringRun:
    """Load once, predict the whole batch in one vectorised call."""
    load, calls = _counting_loader(model_uri)
    model = load()
    return ScoringRun(predictions=[float(value) for value in model.predict(frame)], loads=calls[0])


def wrong_renamed_columns(model_uri: str, frame: pd.DataFrame) -> Any:
    """⚠️ WRONG: score data whose column names differ from the model signature (schema enforcement fails)."""
    renamed = frame.rename(columns={"tenure_months": "tenure"})
    return mlflow.pyfunc.load_model(model_uri).predict(renamed)


def correct_matching_columns(model_uri: str, frame: pd.DataFrame) -> Any:
    """Pass the columns named in the signature (extra columns are ignored, missing ones are errors)."""
    return mlflow.pyfunc.load_model(model_uri).predict(frame)
