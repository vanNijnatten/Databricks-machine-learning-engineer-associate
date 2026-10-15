"""Chapter 7 pitfalls, each as a wrong and a correct version (constitution Principle V).

The ``wrong_*`` functions are deliberately incorrect — never copy them.
"""

import numpy as np
import pandas as pd

from db_ml_associate.common import CHURN_LABEL, DEFAULT_SEED, split_frame


def wrong_mean_impute_skewed(churn: pd.DataFrame, column: str) -> float:
    """⚠️ WRONG: fill a right-skewed column with its mean. Returns the fill value."""
    return float(churn[column].mean())  # pulled up by the long tail


def correct_median_impute(churn: pd.DataFrame, column: str) -> float:
    """Fill a skewed column with its median, which the tail does not move."""
    return float(churn[column].median())


def wrong_impute_before_split(churn: pd.DataFrame, column: str, seed: int = DEFAULT_SEED) -> float:
    """⚠️ WRONG: compute the fill value on all rows, then split. Test rows shaped the training data."""
    # --8<-- [start:wrong_impute_before_split]
    fill_value = churn[column].median()  # includes rows that will become test data
    imputed = churn.fillna({column: fill_value})
    train, test = split_frame(imputed, CHURN_LABEL, seed=seed)
    # --8<-- [end:wrong_impute_before_split]
    return float(fill_value)


def correct_impute_after_split(churn: pd.DataFrame, column: str, seed: int = DEFAULT_SEED) -> float:
    """Split first; learn the fill value from training rows; apply the same value to test rows."""
    # --8<-- [start:correct_impute_after_split]
    train, test = split_frame(churn, CHURN_LABEL, seed=seed)
    fill_value = train[column].median()  # training rows only
    train, test = train.fillna({column: fill_value}), test.fillna({column: fill_value})
    # --8<-- [end:correct_impute_after_split]
    return float(fill_value)


def wrong_one_hot_high_cardinality_for_trees(churn: pd.DataFrame, column: str) -> int:
    """⚠️ WRONG: one-hot encode a column with ~800 categories for a tree model. Returns the new width."""
    return pd.get_dummies(churn[[column]], columns=[column]).shape[1]


def correct_index_high_cardinality(churn: pd.DataFrame, column: str) -> int:
    """Encode categories as one integer index (fine for trees); or target/frequency encoding."""
    codes = churn[column].astype("category").cat.codes.to_frame(f"{column}_index")
    return codes.shape[1]


def wrong_log_of_zero(churn: pd.DataFrame, column: str) -> np.ndarray:
    """⚠️ WRONG: ``log`` of a column that contains zeros produces -inf."""
    return np.log(churn[column].to_numpy(dtype=float))


def correct_log1p(churn: pd.DataFrame, column: str) -> np.ndarray:
    """``log1p(x) = log(1 + x)`` is finite at 0 and undone with ``expm1``."""
    return np.log1p(churn[column].to_numpy(dtype=float))
