"""Chapter 6 pitfall: NaN is not null in Spark (constitution Principle V).

When a pandas DataFrame with missing floats becomes a Spark DataFrame, the gaps arrive as NaN
("not a number", a valid double value) instead of SQL null. Spark counts NaN as a value, sorts it
above every number and propagates it through mean and stddev, so ``summary()`` silently breaks.

The ``wrong_*`` function is deliberately incorrect — never copy it.
"""

from collections.abc import Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def wrong_summary_with_nan(frame: DataFrame, column: str) -> dict[str, str]:
    """⚠️ WRONG: summarise a column whose missing values are NaN. Mean and max come out as NaN."""
    rows = frame.select(column).summary("count", "mean", "max").collect()
    return {row["summary"]: row[column] for row in rows}


def nan_to_null(frame: DataFrame, columns: Sequence[str]) -> DataFrame:
    """Replace NaN by null in the given floating-point columns."""
    # --8<-- [start:nan_to_null]
    for column in columns:
        frame = frame.withColumn(column, F.when(F.isnan(column), None).otherwise(F.col(column)))
    # --8<-- [end:nan_to_null]
    return frame


def correct_summary_after_nan_to_null(frame: DataFrame, column: str) -> dict[str, str]:
    """Convert NaN to null first: count excludes missing rows and mean/max are real numbers."""
    rows = nan_to_null(frame, [column]).select(column).summary("count", "mean", "max").collect()
    return {row["summary"]: row[column] for row in rows}
