"""Feature engineering client calls (S1.06-S1.10) against a fake FeatureEngineeringClient."""

from unittest.mock import MagicMock

import mlflow
import pytest
from databricks.feature_engineering import FeatureLookup

from db_ml_associate.ch03_features.workspace import (
    ADD_PRIMARY_KEY_SQL,
    CREATE_FEATURE_TABLE_SQL,
    build_training_set,
    create_customer_feature_table,
    create_online_store,
    log_feature_model,
    publish_online,
    score,
    upsert_features,
)

TABLE = "ml_associate.tutorial.customer_features"


def test_create_table_with_primary_key() -> None:
    fe, df = MagicMock(), MagicMock()
    create_customer_feature_table(fe, TABLE, df)
    kwargs = fe.create_table.call_args.kwargs
    assert kwargs["name"] == TABLE
    assert kwargs["primary_keys"] == ["customer_id"]
    assert kwargs["df"] is df
    assert kwargs["description"]


def test_write_merges() -> None:
    fe, df = MagicMock(), MagicMock()
    upsert_features(fe, TABLE, df)
    fe.write_table.assert_called_once_with(name=TABLE, df=df, mode="merge")


def test_training_set_uses_a_feature_lookup_on_the_key() -> None:
    fe, labels = MagicMock(), MagicMock()
    training_set, training_df = build_training_set(fe, labels, TABLE)
    assert training_set is fe.create_training_set.return_value
    assert training_df is training_set.load_df.return_value
    kwargs = fe.create_training_set.call_args.kwargs
    (lookup,) = kwargs["feature_lookups"]
    assert isinstance(lookup, FeatureLookup)
    assert lookup.table_name == TABLE
    assert lookup.lookup_key == "customer_id"
    assert kwargs["label"] == "churned"
    assert kwargs["exclude_columns"] == ["customer_id"]
    assert kwargs["df"] is labels


def test_log_model_packages_the_training_set() -> None:
    fe, model, training_set = MagicMock(), MagicMock(), MagicMock()
    log_feature_model(fe, model, training_set, "ml_associate.tutorial.churn_model")
    kwargs = fe.log_model.call_args.kwargs
    assert kwargs["model"] is model
    assert kwargs["training_set"] is training_set
    assert kwargs["flavor"] is mlflow.sklearn
    assert kwargs["registered_model_name"] == "ml_associate.tutorial.churn_model"
    assert kwargs["infer_input_example"] is True


def test_score_batch_looks_up_features() -> None:
    fe, df = MagicMock(), MagicMock()
    score(fe, "models:/ml_associate.tutorial.churn_model@champion", df)
    fe.score_batch.assert_called_once_with(model_uri="models:/ml_associate.tutorial.churn_model@champion", df=df)


def test_online_store_and_publish() -> None:
    fe = MagicMock()
    store = create_online_store(fe, "churn-store")
    fe.create_online_store.assert_called_once_with(name="churn-store", capacity="CU_1")
    publish_online(fe, store, TABLE, f"{TABLE}_online")
    kwargs = fe.publish_table.call_args.kwargs
    assert kwargs == {
        "online_store": fe.create_online_store.return_value,
        "source_table_name": TABLE,
        "online_table_name": f"{TABLE}_online",
        "streaming": False,
    }


def test_sql_alternatives_declare_primary_keys() -> None:
    assert "PRIMARY KEY (customer_id)" in CREATE_FEATURE_TABLE_SQL
    assert "NOT NULL" in CREATE_FEATURE_TABLE_SQL
    assert "ADD CONSTRAINT" in ADD_PRIMARY_KEY_SQL


@pytest.mark.databricks
def test_feature_table_round_trip_in_workspace() -> None:  # pragma: no cover - workspace only
    from databricks.feature_engineering import FeatureEngineeringClient
    from pyspark.sql import SparkSession

    from db_ml_associate.ch03_features.features import compute_customer_features
    from db_ml_associate.datasets import load_customer_churn

    spark = SparkSession.builder.getOrCreate()
    fe = FeatureEngineeringClient()
    name = "ml_associate.tests.customer_features"
    features = compute_customer_features(spark.createDataFrame(load_customer_churn()))
    create_customer_feature_table(fe, name, features)
    assert fe.read_table(name=name).count() == 5000
    fe.drop_table(name=name)
