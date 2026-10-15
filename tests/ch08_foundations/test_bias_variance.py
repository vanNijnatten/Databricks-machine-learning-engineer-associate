"""Model complexity and the bias-variance trade-off (S3.15)."""

import pandas as pd
import pytest

from db_ml_associate.ch08_foundations.bias_variance import complexity_sweep
from db_ml_associate.common import HOUSE_TARGET, split_frame
from db_ml_associate.datasets import load_house_prices


@pytest.fixture(scope="module")
def sweep() -> pd.DataFrame:
    train, validation = split_frame(load_house_prices(), HOUSE_TARGET, stratify=False)
    return complexity_sweep(train, validation, depths=(1, 3, 6, 20)).set_index("depth")


def test_shallow_tree_underfits(sweep: pd.DataFrame) -> None:
    # High bias: both errors are high and close together.
    assert sweep.loc[1, "train_rmse"] > sweep.loc[6, "train_rmse"]
    assert sweep.loc[1, "validation_rmse"] > sweep.loc[6, "validation_rmse"]
    assert sweep.loc[1, "validation_rmse"] - sweep.loc[1, "train_rmse"] < 0.05


def test_deep_tree_overfits(sweep: pd.DataFrame) -> None:
    # High variance: almost no training error, a large gap to validation error.
    assert sweep.loc[20, "train_rmse"] < 0.05
    gap_deep = sweep.loc[20, "validation_rmse"] - sweep.loc[20, "train_rmse"]
    gap_shallow = sweep.loc[3, "validation_rmse"] - sweep.loc[3, "train_rmse"]
    assert gap_deep > 3 * gap_shallow


def test_training_error_falls_monotonically_with_depth(sweep: pd.DataFrame) -> None:
    assert sweep["train_rmse"].is_monotonic_decreasing
