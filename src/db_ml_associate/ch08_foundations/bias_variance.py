"""Model complexity and the bias-variance trade-off (exam objective S3.15).

A decision tree's ``max_depth`` controls its complexity. Shallow trees underfit (high bias: high
error on both training and validation data); very deep trees overfit (high variance: almost no
training error but a large gap to the validation error).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from db_ml_associate.common import DEFAULT_SEED, HOUSE_NUMERIC, HOUSE_TARGET


def complexity_sweep(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    depths: Sequence[int] = (1, 2, 3, 4, 6, 8, 12, 20),
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Fit one tree per depth on log(price) and return train and validation RMSE (log units)."""
    medians = train[list(HOUSE_NUMERIC)].median()  # learned from training data only
    x_train = train[list(HOUSE_NUMERIC)].fillna(medians)
    x_validation = validation[list(HOUSE_NUMERIC)].fillna(medians)
    y_train, y_validation = (
        np.log(train[HOUSE_TARGET]),
        np.log(validation[HOUSE_TARGET]),
    )
    rows = []
    # --8<-- [start:sweep]
    for depth in depths:
        tree = DecisionTreeRegressor(max_depth=depth, random_state=seed).fit(x_train, y_train)
        rows.append(
            {
                "depth": depth,
                "train_rmse": root_mean_squared_error(y_train, tree.predict(x_train)),
                "validation_rmse": root_mean_squared_error(y_validation, tree.predict(x_validation)),
            }
        )
    # --8<-- [end:sweep]
    return pd.DataFrame(rows)
