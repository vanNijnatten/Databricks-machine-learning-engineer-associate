"""Feature engineering in Unity Catalog with ``FeatureEngineeringClient`` (S1.06-S1.10).

Requires a Unity Catalog workspace. The client is passed in (``fe = FeatureEngineeringClient()``) so
the code can be tested without a workspace. Signatures follow ``databricks-feature-engineering``
0.13.0.1, the version preinstalled in Databricks Runtime 18 LTS ML.
"""

from pathlib import Path
from typing import Any

import mlflow
from databricks.feature_engineering import FeatureLookup

from db_ml_associate.common import CHURN_KEY, CHURN_LABEL

SQL_DIR = Path(__file__).with_name("sql")
CREATE_FEATURE_TABLE_SQL = (SQL_DIR / "create_feature_table.sql").read_text(encoding="utf-8")
ADD_PRIMARY_KEY_SQL = (SQL_DIR / "add_primary_key.sql").read_text(encoding="utf-8")


def create_customer_feature_table(fe: Any, name: str, features: Any) -> Any:
    """Create a Unity Catalog feature table from a Spark DataFrame."""
    # --8<-- [start:create_table]
    return fe.create_table(
        name=name,  # catalog.schema.table in Unity Catalog
        primary_keys=[CHURN_KEY],  # required: the key every lookup uses
        df=features,  # schema and initial data; or schema=... for an empty table
        description="Customer behaviour features for churn models",
    )
    # --8<-- [end:create_table]


def upsert_features(fe: Any, name: str, features: Any) -> None:
    """Write new or changed feature rows into an existing feature table."""
    # --8<-- [start:write_table]
    fe.write_table(name=name, df=features, mode="merge")  # upsert by primary key
    # --8<-- [end:write_table]


def build_training_set(fe: Any, labels: Any, table_name: str) -> tuple[Any, Any]:
    """Join features from the feature table to keys and labels; return the TrainingSet and its DataFrame."""
    # --8<-- [start:training_set]
    lookup = FeatureLookup(
        table_name=table_name,
        lookup_key=CHURN_KEY,  # column in `labels` that matches the table's primary key
        feature_names=None,  # None = every feature column of the table
    )
    training_set = fe.create_training_set(
        df=labels,  # customer_id + churned
        feature_lookups=[lookup],
        label=CHURN_LABEL,
        exclude_columns=[CHURN_KEY],  # keys are not features
    )
    training_df = training_set.load_df()  # Spark DataFrame: features + label → train any model on it
    # --8<-- [end:training_set]
    return training_set, training_df


def log_feature_model(fe: Any, model: Any, training_set: Any, registered_model_name: str) -> None:
    """Log a model together with its feature lookups so scoring can fetch features automatically."""
    # --8<-- [start:log_model]
    fe.log_model(
        model=model,
        artifact_path="model",  # the feature engineering client still uses artifact_path
        flavor=mlflow.sklearn,
        training_set=training_set,  # records which features to look up at inference
        registered_model_name=registered_model_name,  # e.g. "ml_associate.tutorial.churn_model"
        infer_input_example=True,
    )
    # --8<-- [end:log_model]


def score(fe: Any, model_uri: str, keys: Any) -> Any:
    """Batch scoring: pass only keys (and non-feature-table columns); features are looked up."""
    # --8<-- [start:score_batch]
    predictions = fe.score_batch(model_uri=model_uri, df=keys)  # adds a `prediction` column
    # --8<-- [end:score_batch]
    return predictions


def create_online_store(fe: Any, name: str, capacity: str = "CU_1") -> Any:
    """Provision a Lakebase-backed online feature store (paid feature; delete it when done)."""
    # --8<-- [start:online_store]
    return fe.create_online_store(name=name, capacity=capacity)  # CU_1, CU_2, CU_4 or CU_8
    # --8<-- [end:online_store]


def publish_online(fe: Any, online_store: Any, source_table_name: str, online_table_name: str) -> Any:
    """Sync an offline feature table to the online store for low-latency lookups."""
    # --8<-- [start:publish_table]
    # Requires a primary key and Change Data Feed on the source table:
    # ALTER TABLE <source> SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
    return fe.publish_table(
        online_store=online_store,
        source_table_name=source_table_name,
        online_table_name=online_table_name,
        streaming=False,  # False → TRIGGERED sync; True → CONTINUOUS sync
    )
    # --8<-- [end:publish_table]
