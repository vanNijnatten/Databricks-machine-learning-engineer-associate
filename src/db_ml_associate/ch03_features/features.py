"""Computing the customer features that go into a feature table (exam objectives S1.06, S1.07).

A feature table holds *features* keyed by an entity ID; labels live elsewhere (here: a table with
``customer_id`` and ``churned``) and are joined to the features when a training set is built.
"""

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from db_ml_associate.common import CHURN_KEY, CHURN_LABEL

FEATURE_COLUMNS = (
    "tenure_months",
    "monthly_charges",
    "data_usage_gb",
    "support_calls",
    "support_calls_per_year",
    "is_month_to_month",
    "payment_method",
)


def compute_customer_features(customers: DataFrame) -> DataFrame:
    """Spark: one row per customer with raw and derived features (no label)."""
    # --8<-- [start:compute_features]
    return customers.select(
        CHURN_KEY,  # primary key of the feature table
        "tenure_months",
        "monthly_charges",
        "data_usage_gb",
        "support_calls",
        (F.col("support_calls") / F.greatest(F.col("tenure_months") / 12, F.lit(1.0))).alias("support_calls_per_year"),
        (F.col("contract_type") == "month-to-month").cast("int").alias("is_month_to_month"),
        "payment_method",
    )
    # --8<-- [end:compute_features]


def compute_customer_features_pandas(customers: pd.DataFrame) -> pd.DataFrame:
    """pandas equivalent of :func:`compute_customer_features`."""
    features = customers[[CHURN_KEY, "tenure_months", "monthly_charges", "data_usage_gb", "support_calls"]].copy()
    features["support_calls_per_year"] = customers["support_calls"] / (customers["tenure_months"] / 12).clip(lower=1)
    features["is_month_to_month"] = (customers["contract_type"] == "month-to-month").astype(int)
    features["payment_method"] = customers["payment_method"]
    return features


def label_frame(customers: pd.DataFrame) -> pd.DataFrame:
    """The labels for training: the lookup key and the label only."""
    return customers[[CHURN_KEY, CHURN_LABEL]].copy()
