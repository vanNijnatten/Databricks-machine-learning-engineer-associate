"""Chapter 11 pitfall: evaluating a log-target model on the log scale (exam objective S3.14).

The ``wrong_*`` function is deliberately incorrect — never copy it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

from db_ml_associate.common import (
    DEFAULT_SEED,
    HOUSE_NUMERIC,
    HOUSE_TARGET,
    split_frame,
)


@dataclass(frozen=True)
class PriceEvaluation:
    """RMSE in the target's own unit and the back-transformed predictions."""

    rmse: float
    predictions: np.ndarray


def _fit_log_model(houses: pd.DataFrame, seed: int) -> tuple[LinearRegression, pd.DataFrame, pd.DataFrame]:
    train, test = split_frame(houses, HOUSE_TARGET, seed=seed, stratify=False)
    medians = train[list(HOUSE_NUMERIC)].median()
    x_train, x_test = (
        train[list(HOUSE_NUMERIC)].fillna(medians),
        test[list(HOUSE_NUMERIC)].fillna(medians),
    )
    model = LinearRegression().fit(x_train, np.log(train[HOUSE_TARGET]))  # trained on log(price)
    return model, x_test, test


def wrong_rmse_on_log_scale(houses: pd.DataFrame, seed: int = DEFAULT_SEED) -> float:
    """⚠️ WRONG: compute RMSE on log(price). The number is in log units, not euros."""
    model, x_test, test = _fit_log_model(houses, seed)
    # --8<-- [start:wrong_rmse_on_log_scale]
    log_predictions = model.predict(x_test)
    rmse = root_mean_squared_error(np.log(test[HOUSE_TARGET]), log_predictions)  # ≈ 0.37 "log euros"
    # --8<-- [end:wrong_rmse_on_log_scale]
    return float(rmse)


def correct_exponentiate_before_metrics(houses: pd.DataFrame, seed: int = DEFAULT_SEED) -> PriceEvaluation:
    """Exponentiate predictions back to prices, then compute RMSE in euros."""
    model, x_test, test = _fit_log_model(houses, seed)
    # --8<-- [start:correct_exponentiate_before_metrics]
    predicted_prices = np.exp(model.predict(x_test))  # undo np.log (use np.expm1 to undo np.log1p)
    rmse = root_mean_squared_error(test[HOUSE_TARGET], predicted_prices)  # in euros
    # --8<-- [end:correct_exponentiate_before_metrics]
    return PriceEvaluation(rmse=float(rmse), predictions=predicted_prices)
