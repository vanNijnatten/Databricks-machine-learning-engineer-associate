"""Removing outliers from a Spark DataFrame (exam objective S2.02).

Both filters keep rows whose value is missing: a missing value is not an outlier, and it will be
imputed later. Learn the bounds on training data and apply them to new data.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def remove_outliers_stddev(frame: DataFrame, column: str, threshold: float = 3.0) -> DataFrame:
    """Drop rows more than ``threshold`` standard deviations from the mean (z-score filter)."""
    # --8<-- [start:stddev_filter]
    stats = frame.agg(F.mean(column).alias("mean"), F.stddev(column).alias("std")).first()
    lower = stats["mean"] - threshold * stats["std"]
    upper = stats["mean"] + threshold * stats["std"]
    kept = frame.filter(F.col(column).isNull() | F.col(column).between(lower, upper))
    # --8<-- [end:stddev_filter]
    return kept


def iqr_bounds(frame: DataFrame, column: str, factor: float = 1.5, relative_error: float = 0.0) -> tuple[float, float]:
    """Return Tukey's fences ``(Q1 - factor·IQR, Q3 + factor·IQR)``."""
    # --8<-- [start:iqr_bounds]
    q1, q3 = frame.approxQuantile(column, [0.25, 0.75], relative_error)  # 0.0 = exact (costly on big data)
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr
    # --8<-- [end:iqr_bounds]


def remove_outliers_iqr(frame: DataFrame, column: str, factor: float = 1.5) -> DataFrame:
    """Drop rows outside the IQR fences (robust to the outliers themselves)."""
    # --8<-- [start:iqr_filter]
    lower, upper = iqr_bounds(frame, column, factor)
    return frame.filter(F.col(column).isNull() | F.col(column).between(lower, upper))
    # --8<-- [end:iqr_filter]
