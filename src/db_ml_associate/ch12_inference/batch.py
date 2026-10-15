"""Batch inference with pandas, Spark UDFs and ``ai_query`` (exam objectives S4.01, S4.03)."""

from collections.abc import Sequence
from pathlib import Path

import mlflow
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def batch_predict_pandas(model_uri: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Score a pandas DataFrame on the driver and return it with a ``prediction`` column."""
    # --8<-- [start:pandas_batch]
    model = mlflow.pyfunc.load_model(model_uri)  # load ONCE, e.g. "models:/prod.ml.churn@champion"
    scored = frame.copy()
    scored["prediction"] = model.predict(frame)  # the signature selects and checks the input columns
    # --8<-- [end:pandas_batch]
    return scored


def batch_predict_spark(spark: SparkSession, model_uri: str, frame: DataFrame, features: Sequence[str]) -> DataFrame:
    """Score a Spark DataFrame in parallel on the executors with an MLflow Spark UDF."""
    # --8<-- [start:spark_udf]
    churn_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="double")
    scored = frame.withColumn("prediction", churn_model(F.struct(*features)))  # struct keeps column names
    # scored.write.mode("overwrite").saveAsTable("ml_associate.tutorial.churn_scores")
    # --8<-- [end:spark_udf]
    return scored


AI_QUERY_SQL = (Path(__file__).with_name("sql") / "ai_query.sql").read_text(encoding="utf-8")
