"""One-hot encoding of categorical features (exam objectives S2.07, S2.08)."""

from collections.abc import Sequence

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.sql import DataFrame


def one_hot_spark(frame: DataFrame, columns: Sequence[str], keep_unseen: bool = True) -> DataFrame:
    """Index each string column, then one-hot encode the indices into sparse vectors.

    With ``keep_unseen`` the indexer reserves one extra index for categories not seen in training,
    so *k* categories give vectors of size *k* (k + 1 indices, minus the dropped last one);
    without it, size *k - 1* and unseen categories raise an error at transform time.
    """
    # --8<-- [start:one_hot_spark]
    indexed = [f"{column}_index" for column in columns]
    encoded = [f"{column}_vector" for column in columns]
    pipeline = Pipeline(
        stages=[
            StringIndexer(
                inputCols=list(columns), outputCols=indexed, handleInvalid="keep" if keep_unseen else "error"
            ),
            OneHotEncoder(inputCols=indexed, outputCols=encoded),  # dropLast=True drops one slot
        ]
    )
    return pipeline.fit(frame).transform(frame)
    # --8<-- [end:one_hot_spark]


def one_hot_pandas(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """pandas equivalent: one 0/1 column per category."""
    # --8<-- [start:one_hot_pandas]
    return pd.get_dummies(frame, columns=list(columns), dtype=int)  # drop_first=True drops one category
    # --8<-- [end:one_hot_pandas]
