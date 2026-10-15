"""Imputing missing values with the mean, median or mode (exam objectives S2.05, S2.06)."""

from collections.abc import Sequence

import pandas as pd
from pyspark.ml.feature import Imputer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def impute_spark(frame: DataFrame, columns: Sequence[str], strategy: str = "median") -> DataFrame:
    """Fill missing numeric values in place with Spark ML's ``Imputer`` (mean, median or mode)."""
    # --8<-- [start:spark_imputer]
    imputer = Imputer(
        strategy=strategy,  # "mean" (default), "median" or "mode"
        inputCols=list(columns),
        outputCols=list(columns),  # overwrite; use new names to keep the originals
    )  # an estimator: it learns the fill values in fit()
    model = imputer.fit(frame)  # fit on TRAINING data only
    imputed = model.transform(frame)  # treats both null and NaN as missing
    # --8<-- [end:spark_imputer]
    return imputed


def impute_categorical_mode(frame: DataFrame, column: str) -> DataFrame:
    """Fill a string column with its most frequent value (``Imputer`` only handles numbers)."""
    # --8<-- [start:categorical_mode]
    mode = frame.filter(F.col(column).isNotNull()).groupBy(column).count().orderBy(F.desc("count")).first()[column]
    return frame.fillna({column: mode})
    # --8<-- [end:categorical_mode]


def add_missing_indicator(frame: DataFrame, column: str) -> DataFrame:
    """Add ``<column>_was_missing`` (1/0) before imputing, so the model can still see the gap."""
    return frame.withColumn(f"{column}_was_missing", F.col(column).isNull().cast("int"))


def impute_pandas(frame: pd.DataFrame, column: str, strategy: str) -> pd.DataFrame:
    """pandas equivalent: fill with the column's mean, median or mode."""
    # --8<-- [start:pandas_impute]
    fill_values = {
        "mean": frame[column].mean(),
        "median": frame[column].median(),
        "mode": frame[column].mode().iloc[0],  # mode() can return several values; take the first
    }
    return frame.fillna({column: fill_values[strategy]})
    # --8<-- [end:pandas_impute]
