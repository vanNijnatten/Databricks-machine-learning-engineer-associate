"""Streaming inference with Structured Streaming (exam objectives S4.01, S4.04).

The same Spark UDF that scores a batch DataFrame scores a streaming DataFrame; only the source
(``readStream``) and the sink (``writeStream``) change. In production, prefer a Lakeflow pipeline
(see ``churn_scoring_pipeline.py``), which manages checkpoints, retries and autoscaling for you.
"""

from collections.abc import Sequence

import mlflow
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def score_stream(spark: SparkSession, stream: DataFrame, model_uri: str, features: Sequence[str]) -> DataFrame:
    """Add a ``churn_prediction`` column to every micro-batch of a streaming DataFrame."""
    # --8<-- [start:structured_streaming]
    churn_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="double")
    scored = stream.withColumn("churn_prediction", churn_model(F.struct(*features)))
    # (scored.writeStream
    #     .option("checkpointLocation", "/Volumes/ml_associate/tutorial/checkpoints/churn")
    #     .trigger(availableNow=True)        # or processingTime="1 minute" for continuous scoring
    #     .toTable("ml_associate.tutorial.churn_scores_stream"))
    # --8<-- [end:structured_streaming]
    return scored
