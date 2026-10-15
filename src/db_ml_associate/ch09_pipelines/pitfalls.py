"""Chapter 9 pitfalls, each as a wrong and a correct version (constitution Principle V).

The ``wrong_*`` functions are deliberately incorrect — never copy them.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from db_ml_associate.common import CHURN_LABEL, DEFAULT_SEED, split_frame


def wrong_fit_scaler_on_full_data(
    churn: pd.DataFrame, column: str, seed: int = DEFAULT_SEED
) -> tuple[float, np.ndarray, np.ndarray]:
    """⚠️ WRONG: fit the scaler before splitting. Returns the learned mean and the scaled splits."""
    # --8<-- [start:wrong_fit_scaler_on_full_data]
    scaler = StandardScaler().fit(churn[[column]])  # learns from rows that will become test data
    train, test = split_frame(churn, CHURN_LABEL, seed=seed)
    train_scaled, test_scaled = (
        scaler.transform(train[[column]]),
        scaler.transform(test[[column]]),
    )
    # --8<-- [end:wrong_fit_scaler_on_full_data]
    return float(scaler.mean_[0]), train_scaled, test_scaled


def correct_fit_inside_pipeline(churn: pd.DataFrame, column: str, seed: int = DEFAULT_SEED) -> tuple[float, np.ndarray]:
    """Split first; the pipeline fits the scaler on training rows only. Returns the mean and scaled test data."""
    # --8<-- [start:correct_fit_inside_pipeline]
    train, test = split_frame(churn, CHURN_LABEL, seed=seed)
    pipeline = Pipeline([("scale", StandardScaler())]).fit(train[[column]])  # training statistics only
    test_scaled = pipeline.transform(test[[column]])  # reuses the training mean and std
    # --8<-- [end:correct_fit_inside_pipeline]
    return float(pipeline.named_steps["scale"].mean_[0]), test_scaled


def wrong_transform_with_unfitted_transformer(churn: pd.DataFrame, column: str) -> np.ndarray:
    """⚠️ WRONG: call transform() before fit(). Raises ``NotFittedError``."""
    return StandardScaler().transform(churn[[column]])


def correct_fit_then_transform(churn: pd.DataFrame, column: str) -> np.ndarray:
    """Fit first (or use ``fit_transform`` on training data), then transform."""
    return StandardScaler().fit_transform(churn[[column]])
