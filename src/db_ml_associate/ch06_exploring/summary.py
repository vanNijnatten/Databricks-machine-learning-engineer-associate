"""Summary statistics for Spark and pandas DataFrames (exam objective S2.01)."""

from collections.abc import Sequence
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame


def spark_summary(frame: DataFrame, columns: Sequence[str] | None = None, statistics: Sequence[str] = ()) -> DataFrame:
    """Return ``DataFrame.summary()`` for the given columns.

    Without ``statistics`` Spark computes count, mean, stddev, min, 25%, 50%, 75% and max.
    """
    selected = frame.select(*columns) if columns else frame
    return selected.summary(*statistics)


def summary_examples(frame: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    """The three ways to get summary statistics from a Spark DataFrame."""
    # --8<-- [start:spark_summary]
    everything = frame.summary()  # count, mean, stddev, min, 25%, 50%, 75%, max
    chosen = frame.summary("count", "min", "25%", "75%", "max")  # only the statistics you name
    basic = frame.describe()  # count, mean, stddev, min, max: no percentiles
    # --8<-- [end:spark_summary]
    return everything, chosen, basic


def pandas_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """``describe(include="all")``: numeric statistics plus count/unique/top/freq for categoricals."""
    return frame.describe(include="all")


def databricks_data_summary(dbutils: Any, frame: Any, precise: bool = False) -> None:
    """Show Databricks' interactive data summary (workspace only; ``dbutils`` exists in notebooks)."""
    # --8<-- [start:dbutils_summarize]
    dbutils.data.summarize(frame, precise=precise)  # precise=True: exact statistics, slower
    # --8<-- [end:dbutils_summarize]
