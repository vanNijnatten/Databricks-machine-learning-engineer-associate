"""Mitigating class imbalance (exam objective S3.02).

Two families of techniques:

* **Cost-sensitive learning** — give minority-class errors a higher weight during training
  (``class_weight`` in scikit-learn, ``weightCol`` in Spark ML). The data is unchanged.
* **Resampling** — change the training data: duplicate minority rows (oversampling) or drop
  majority rows (undersampling). Only ever resample the *training* split.
"""

from collections.abc import Mapping

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from db_ml_associate.common import CHURN_CATEGORICAL, CHURN_NUMERIC, DEFAULT_SEED


def balanced_class_weights(labels: pd.Series) -> dict[int, float]:
    """Return scikit-learn's "balanced" weights: ``n_samples / (n_classes * count_of_class)``."""
    # --8<-- [start:balanced_weights]
    counts = labels.value_counts()
    n_samples, n_classes = len(labels), len(counts)
    return {int(label): n_samples / (n_classes * count) for label, count in counts.items()}
    # --8<-- [end:balanced_weights]


def churn_preprocessor() -> ColumnTransformer:
    """Impute and scale numeric churn features, one-hot encode the low-cardinality categoricals."""
    return ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                list(CHURN_NUMERIC),
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CHURN_CATEGORICAL),
            ),
        ]
    )


def make_churn_classifier(
    class_weight: str | Mapping[int, float] | None = None,
    estimator: ClassifierMixin | None = None,
) -> Pipeline:
    """A churn classification pipeline; pass ``class_weight="balanced"`` for cost-sensitive learning."""
    # --8<-- [start:class_weight]
    model = estimator or LogisticRegression(class_weight=class_weight, max_iter=1000)
    # --8<-- [end:class_weight]
    return Pipeline([("prepare", churn_preprocessor()), ("model", model)])


def minority_recall(model: Pipeline, features: pd.DataFrame, labels: pd.Series) -> float:
    """Recall of the positive (minority) class: the share of churners the model finds."""
    return float(recall_score(labels, model.predict(features), pos_label=1))


def random_oversample(train: pd.DataFrame, label: str, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Duplicate random minority rows until both classes are equally frequent (training data only)."""
    # --8<-- [start:oversample]
    counts = train[label].value_counts()
    minority = train[train[label] == counts.idxmin()]
    extra = minority.sample(n=counts.max() - counts.min(), replace=True, random_state=seed)
    return pd.concat([train, extra], ignore_index=True)
    # --8<-- [end:oversample]


def random_undersample(train: pd.DataFrame, label: str, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Keep every minority row and an equally sized random sample of the majority (training data only)."""
    counts = train[label].value_counts()
    minority = train[train[label] == counts.idxmin()]
    majority = train[train[label] == counts.idxmax()].sample(n=counts.min(), random_state=seed)
    return pd.concat([minority, majority], ignore_index=True)


def add_spark_class_weights(frame: SparkDataFrame, label: str, weight_col: str = "class_weight") -> SparkDataFrame:
    """Add a balanced per-row weight column for Spark ML estimators that accept ``weightCol``."""
    # --8<-- [start:spark_weight_col]
    n_rows = F.count(F.lit(1)).over(Window.partitionBy())
    n_in_class = F.count(F.lit(1)).over(Window.partitionBy(label))
    n_classes = frame.select(label).distinct().count()
    weighted = frame.withColumn(weight_col, n_rows / (F.lit(n_classes) * n_in_class))
    # Then: LogisticRegression(labelCol=label, weightCol=weight_col)
    # --8<-- [end:spark_weight_col]
    return weighted
