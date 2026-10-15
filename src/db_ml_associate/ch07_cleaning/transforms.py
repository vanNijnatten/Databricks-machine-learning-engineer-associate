"""Log-scale transformation of right-skewed features or targets (exam objective S2.09)."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def log_transform_spark(frame: DataFrame, column: str) -> DataFrame:
    """Add ``<column>_log = log(1 + value)``; undo with ``expm1`` (chapter 11)."""
    # --8<-- [start:log_transform]
    return frame.withColumn(f"{column}_log", F.log1p(column))  # log1p is defined at 0; log(0) is not
    # --8<-- [end:log_transform]
