"""Batch inference with pandas and Spark, streaming scoring, and the pipeline source (S4.01, S4.03, S4.04)."""

import ast
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.models import infer_signature
from pyspark.sql import SparkSession

import db_ml_associate.ch12_inference
from db_ml_associate.ch09_pipelines.sklearn_pipeline import build_sklearn_churn_pipeline
from db_ml_associate.ch12_inference.batch import AI_QUERY_SQL, batch_predict_pandas, batch_predict_spark
from db_ml_associate.ch12_inference.streaming import score_stream
from db_ml_associate.common import CHURN_CATEGORICAL, CHURN_LABEL, CHURN_NUMERIC, split_frame
from db_ml_associate.datasets import load_customer_churn

FEATURES = [*CHURN_NUMERIC, *CHURN_CATEGORICAL]


@pytest.fixture(scope="module")
def churn_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = split_frame(load_customer_churn(), CHURN_LABEL)
    return train, test.head(200)


@pytest.fixture
def model_uri(local_mlflow: str, churn_split: tuple[pd.DataFrame, pd.DataFrame]) -> str:
    train, _ = churn_split
    features = train[FEATURES]
    model = build_sklearn_churn_pipeline().fit(features, train[CHURN_LABEL])
    with mlflow.start_run():
        info = mlflow.sklearn.log_model(
            model, name="model", signature=infer_signature(features, model.predict(features))
        )
    return info.model_uri


def test_pandas_batch_inference_aligns_with_input(
    model_uri: str, churn_split: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    _, test = churn_split
    scored = batch_predict_pandas(model_uri, test)
    assert len(scored) == len(test)
    assert scored["customer_id"].tolist() == test["customer_id"].tolist()
    assert set(scored["prediction"]) <= {0, 1}


@pytest.mark.spark
def test_spark_udf_matches_pandas(
    spark: SparkSession, model_uri: str, churn_split: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    _, test = churn_split
    pandas_scores = batch_predict_pandas(model_uri, test).set_index("customer_id")["prediction"]
    spark_scores = (
        batch_predict_spark(spark, model_uri, spark.createDataFrame(test), FEATURES)
        .toPandas()
        .set_index("customer_id")["prediction"]
    )
    np.testing.assert_array_equal(spark_scores.loc[pandas_scores.index].to_numpy(), pandas_scores.to_numpy())


@pytest.mark.spark
def test_streaming_scoring(
    spark: SparkSession, model_uri: str, churn_split: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    _, test = churn_split
    source = tmp_path / "events"
    spark.createDataFrame(test).write.parquet(str(source))
    stream = spark.readStream.schema(spark.read.parquet(str(source)).schema).parquet(str(source))
    query = (
        score_stream(spark, stream, model_uri, FEATURES)
        .writeStream.format("memory")
        .queryName("churn_scores")
        .option("checkpointLocation", str(tmp_path / "checkpoint"))
        .start()
    )
    query.processAllAvailable()
    query.stop()
    assert spark.table("churn_scores").count() == len(test)
    assert "churn_prediction" in spark.table("churn_scores").columns


def test_ai_query_example_calls_a_serving_endpoint() -> None:
    assert "ai_query(" in AI_QUERY_SQL
    assert "named_struct(" in AI_QUERY_SQL
    assert "returnType" in AI_QUERY_SQL


def test_pipeline_source_defines_a_streaming_table_with_a_spark_udf() -> None:
    # Pipeline source files run inside a pipeline, so read the file instead of importing it.
    source = (
        Path(db_ml_associate.ch12_inference.__file__).with_name("churn_scoring_pipeline.py").read_text(encoding="utf-8")
    )
    tree = ast.parse(source)
    assert "from pyspark import pipelines as dp" in source
    decorators = {
        ast.unparse(decorator.func if isinstance(decorator, ast.Call) else decorator)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
    }
    assert "dp.table" in decorators
    assert "spark.readStream.table" in source
    assert "mlflow.pyfunc.spark_udf" in source
